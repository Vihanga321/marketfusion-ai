#property strict

input int InpSnapshotSeconds = 300;
input int InpLookaheadHours = 48;
input string InpCountryCode = "US";
input string InpCurrency = "USD";
input string InpOutputFile = "marketfusion_mt5_calendar_snapshots.tsv";

string TimeField(const datetime value)
  {
   if(value==0)
      return "";
   return TimeToString(value,TIME_DATE|TIME_SECONDS);
  }

string UtcIso(const datetime value)
  {
   MqlDateTime dt;
   TimeToStruct(value,dt);
   return StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",dt.year,dt.mon,dt.day,dt.hour,dt.min,dt.sec);
  }

string NumericField(const MqlCalendarValue &value,const int field,const int digits)
  {
   double number=0.0;
   bool available=false;
   if(field==0)
     {
      available=value.HasActualValue();
      if(available) number=value.GetActualValue();
     }
   else if(field==1)
     {
      available=value.HasForecastValue();
      if(available) number=value.GetForecastValue();
     }
   else if(field==2)
     {
      available=value.HasPreviousValue();
      if(available) number=value.GetPreviousValue();
     }
   else if(field==3)
     {
      available=value.HasRevisedValue();
      if(available) number=value.GetRevisedValue();
     }
   if(!available)
      return "";
   return DoubleToString(number,digits);
  }

int OpenAppendFile()
  {
   ResetLastError();
   int handle=FileOpen(InpOutputFile,FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_SHARE_READ,'\t');
   if(handle==INVALID_HANDLE)
      handle=FileOpen(InpOutputFile,FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_SHARE_READ,'\t');
   if(handle==INVALID_HANDLE)
      return INVALID_HANDLE;
   if(FileSize(handle)==0)
     {
      FileWrite(handle,
                "value_id","event_id","event_time_server","reference_period_server","revision",
                "actual_value","forecast_value","prev_value","revised_prev_value","impact_type",
                "event_name","event_code","country_code","currency","unit","importance",
                "multiplier","digits","time_mode","sector","frequency","source_url",
                "exported_at_server","captured_at_gmt");
     }
   FileSeek(handle,0,SEEK_END);
   return handle;
  }

void CaptureSnapshot()
  {
   datetime captured_server=TimeTradeServer();
   datetime captured_gmt=TimeGMT();
   datetime to_time=captured_server+(InpLookaheadHours*3600);

   MqlCalendarValue values[];
   ResetLastError();
   int total=CalendarValueHistory(values,captured_server,to_time,InpCountryCode,InpCurrency);
   if(total<0)
     {
      PrintFormat("MarketFusion snapshot CalendarValueHistory failed. Error=%d",GetLastError());
      return;
     }

   int handle=OpenAppendFile();
   if(handle==INVALID_HANDLE)
     {
      PrintFormat("MarketFusion snapshot FileOpen failed for %s. Error=%d",InpOutputFile,GetLastError());
      return;
     }

   int written=0;
   for(int i=0;i<total;i++)
     {
      MqlCalendarEvent event;
      MqlCalendarCountry country;
      ZeroMemory(event);
      ZeroMemory(country);
      bool have_event=CalendarEventById(values[i].event_id,event);
      bool have_country=false;
      if(have_event)
         have_country=CalendarCountryById(event.country_id,country);

      string event_name=have_event ? event.name : "";
      string event_code=have_event ? event.event_code : "";
      string country_code=have_country ? country.code : InpCountryCode;
      string currency=have_country ? country.currency : InpCurrency;
      string unit=have_event ? EnumToString(event.unit) : "";
      string importance=have_event ? EnumToString(event.importance) : "";
      string multiplier=have_event ? EnumToString(event.multiplier) : "";
      string time_mode=have_event ? EnumToString(event.time_mode) : "";
      string sector=have_event ? EnumToString(event.sector) : "";
      string frequency=have_event ? EnumToString(event.frequency) : "";
      string source_url=have_event ? event.source_url : "";
      int digits=have_event ? (int)event.digits : 8;
      if(digits<0) digits=0;
      if(digits>8) digits=8;

      FileWrite(handle,
                StringFormat("%I64u",values[i].id),
                StringFormat("%I64u",values[i].event_id),
                TimeField(values[i].time),
                TimeField(values[i].period),
                IntegerToString(values[i].revision),
                NumericField(values[i],0,digits),
                NumericField(values[i],1,digits),
                NumericField(values[i],2,digits),
                NumericField(values[i],3,digits),
                EnumToString(values[i].impact_type),
                event_name,event_code,country_code,currency,unit,importance,multiplier,
                IntegerToString(digits),time_mode,sector,frequency,source_url,
                TimeField(captured_server),UtcIso(captured_gmt));
      written++;
     }

   FileFlush(handle);
   FileClose(handle);
   PrintFormat("MarketFusion calendar snapshot: %d rows captured at %s GMT",written,UtcIso(captured_gmt));
  }

int OnInit()
  {
   if(InpSnapshotSeconds<60 || InpLookaheadHours<1)
      return INIT_PARAMETERS_INCORRECT;
   EventSetTimer(InpSnapshotSeconds);
   CaptureSnapshot();
   Print("MarketFusionCalendarSnapshotEA started. It only reads the economic calendar and writes snapshots; it does not trade.");
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
  }

void OnTimer()
  {
   CaptureSnapshot();
  }
