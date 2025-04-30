import * as React from "react";
import { format } from "date-fns";
import { Calendar as CalendarIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function DateRangePicker({ dateRange, onSelect, className }) {
  return (
    <div className={cn("grid gap-2", className)}>
      <div className="grid grid-cols-2 gap-2">
        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant={"outline"}
              className={cn(
                "w-full justify-start text-left font-normal",
                !dateRange.start && "text-muted-foreground"
              )}
            >
              <CalendarIcon className="h-4 w-4 -ml-1" />
              {dateRange.start ? (
                format(dateRange.start, "PPP")
              ) : (
                <span>Start date</span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0">
            <Calendar
              mode="single"
              selected={dateRange.start}
              onSelect={(start) => onSelect({ ...dateRange, start })}
              initialFocus
            />
          </PopoverContent>
        </Popover>

        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant={"outline"}
              className={cn(
                "w-full justify-start text-left font-normal",
                !dateRange.end && "text-muted-foreground"
              )}
            >
              <CalendarIcon className="h-4 w-4 -ml-1" />
              {dateRange.end ? (
                format(dateRange.end, "PPP")
              ) : (
                <span>End date</span>
              )}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0">
            <Calendar
              mode="single"
              selected={dateRange.end}
              onSelect={(end) => onSelect({ ...dateRange, end })}
              initialFocus
            />
          </PopoverContent>
        </Popover>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <Select
          onValueChange={(value) => {
            const end = new Date();
            let start = new Date();

            if (value === "7d") start.setDate(start.getDate() - 7);
            else if (value === "30d") start.setDate(start.getDate() - 30);
            else if (value === "90d") start.setDate(start.getDate() - 90);
            else if (value === "1y") start.setFullYear(start.getFullYear() - 1);
            else if (value === "3y") start.setFullYear(start.getFullYear() - 3);
            else if (value === "5y") start.setFullYear(start.getFullYear() - 5);

            onSelect({ start, end });
          }}
        >
          <SelectTrigger>
            <SelectValue placeholder="Quick select" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="7d">Last 7 days</SelectItem>
            <SelectItem value="30d">Last 30 days</SelectItem>
            <SelectItem value="90d">Last 90 days</SelectItem>
            <SelectItem value="1y">Last year</SelectItem>
            <SelectItem value="3y">Last 3 years</SelectItem>
            <SelectItem value="5y">Last 5 years</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
