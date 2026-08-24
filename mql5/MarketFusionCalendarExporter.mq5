#property strict
#property script_show_inputs

input datetime InpFrom = D'2015.01.01 00:00';
input datetime InpTo = 0;
input string InpCountryCode = "US";
input string InpCurrency = "USD";
input string InpOutputFile = "marketfusion_mt5_calendar_history.tsv";

string TimeField(const datetime value)
  {
   if(value==0)
      return "";
   return TimeToString(value,TIME_DATE|TIME_SECONDS);
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

void OnStart()
  {
   if(InpTo!=0 && InpTo<=InpFrom)
     {
      Print("MarketFusion calendar exporter: InpTo must be later than InpFrom or 0.");
      return;
     }

   MqlCalendarValue values[];
   ResetLastError();
   int total=CalendarValueHistory(values,InpFrom,InpTo,InpCountryCode,InpCurrency);
   if(total<0)
     {
      PrintFormat("CalendarValueHistory failed. Error=%d",GetLastError());
      return;
     }
   if(total==0)
     {
      Print("CalendarValueHistory returned no rows.");
      return;
     }

   ResetLastError();
   int handle=FileOpen(InpOutputFile,FILE_WRITE|FILE_CSV|FILE_ANSI,'\t');
   if(handle==INVALID_HANDLE)
     {
      PrintFormat("FileOpen failed for %s. Error=%d",InpOutputFile,GetLastError());
      return;
     }

   FileWrite(handle,
             "value_id","event_id","event_time_server","reference_period_server","revision",
             "actual_value","forecast_value","prev_value","revised_prev_value","impact_type",
             "event_name","event_code","country_code","currency","unit","importance",
             "multiplier","digits","time_mode","sector","frequency","source_url",
             "exported_at_server");

   datetime exported_at=TimeTradeServer();
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
                TimeField(exported_at));
      written++;
     }

   FileFlush(handle);
   FileClose(handle);
   PrintFormat("MarketFusion historical calendar export complete: %d rows -> MQL5/Files/%s",written,InpOutputFile);
   Print("Important: MT5 calendar times are trade-server time. Do not treat them as UTC without an independent event-day offset audit.");
  }
