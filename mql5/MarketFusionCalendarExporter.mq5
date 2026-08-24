#property strict
#property script_show_inputs

input datetime InpFrom = D'2015.01.01 00:00';
input datetime InpTo = 0;
input string InpCountryCode = "US";
input string InpCurrency = "USD";
input string InpOutputFile = "marketfusion_mt5_calendar_history.tsv";
input int InpBatchDays = 30;

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

bool WriteCalendarRow(const int handle,const MqlCalendarValue &value,const datetime exported_at)
  {
   MqlCalendarEvent event;
   MqlCalendarCountry country;
   ZeroMemory(event);
   ZeroMemory(country);
   bool have_event=CalendarEventById(value.event_id,event);
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

   int fields=FileWrite(handle,
                        StringFormat("%I64u",value.id),
                        StringFormat("%I64u",value.event_id),
                        TimeField(value.time),
                        TimeField(value.period),
                        IntegerToString(value.revision),
                        NumericField(value,0,digits),
                        NumericField(value,1,digits),
                        NumericField(value,2,digits),
                        NumericField(value,3,digits),
                        EnumToString(value.impact_type),
                        event_name,event_code,country_code,currency,unit,importance,multiplier,
                        IntegerToString(digits),time_mode,sector,frequency,source_url,
                        TimeField(exported_at));
   return fields>0;
  }

void OnStart()
  {
   datetime effective_to=(InpTo==0 ? TimeTradeServer() : InpTo);
   if(effective_to<=InpFrom)
     {
      Print("MarketFusion calendar exporter: end time must be later than start time.");
      return;
     }
   if(InpBatchDays<1 || InpBatchDays>90)
     {
      Print("MarketFusion calendar exporter: InpBatchDays must be between 1 and 90.");
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
   datetime cursor=InpFrom;
   int written=0;
   int batches=0;
   bool failed=false;

   while(cursor<=effective_to)
     {
      datetime chunk_to=cursor + InpBatchDays*86400 - 1;
      if(chunk_to>effective_to)
         chunk_to=effective_to;

      MqlCalendarValue values[];
      ArrayFree(values);
      ResetLastError();
      int total=CalendarValueHistory(values,cursor,chunk_to,InpCountryCode,InpCurrency);
      if(total<0)
        {
         int error=GetLastError();
         PrintFormat("CalendarValueHistory failed for %s -> %s. Error=%d",
                     TimeField(cursor),TimeField(chunk_to),error);
         failed=true;
         break;
        }

      for(int i=0;i<total;i++)
        {
         if(!WriteCalendarRow(handle,values[i],exported_at))
           {
            PrintFormat("FileWrite failed after %d rows. Error=%d",written,GetLastError());
            failed=true;
            break;
           }
         written++;
        }
      if(failed)
         break;

      batches++;
      if((batches%12)==0 || chunk_to>=effective_to)
         PrintFormat("MarketFusion calendar export progress: batches=%d rows=%d through %s",
                     batches,written,TimeField(chunk_to));

      if(chunk_to>=effective_to)
         break;
      cursor=chunk_to+1;
     }

   FileFlush(handle);
   FileClose(handle);

   if(failed)
     {
      FileDelete(InpOutputFile);
      Print("MarketFusion historical calendar export FAILED; partial output deleted.");
      return;
     }

   PrintFormat("MarketFusion historical calendar export complete: %d rows in %d batches -> MQL5/Files/%s",
               written,batches,InpOutputFile);
   Print("Important: MT5 calendar times are trade-server time. Do not treat them as UTC without an independent event-day offset audit.");
  }
