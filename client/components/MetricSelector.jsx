import * as React from "react";
import { Check, ChevronsUpDown } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

const metrics = [
  { value: "stock_price", label: "Stock Price" },
  { value: "it_spending", label: "IT Spending" },
  { value: "digital_transactions", label: "Digital Transactions" },
  { value: "mobile_users", label: "Mobile Users" },
  { value: "revenue", label: "Revenue" },
  { value: "profit_margin", label: "Profit Margin" },
];

export function MetricSelector({ selectedMetrics, onSelect, className }) {
  const [open, setOpen] = React.useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn("w-full justify-between", className)}
        >
          {selectedMetrics.length > 0
            ? `${selectedMetrics.length} metric(s) selected`
            : "Select metrics..."}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0">
        <Command>
          <CommandInput placeholder="Search metrics..." />
          <CommandEmpty>No metric found.</CommandEmpty>
          <CommandGroup className="max-h-[300px] overflow-y-auto">
            {metrics.map((metric) => (
              <CommandItem
                key={metric.value}
                value={metric.value}
                onSelect={() => {
                  const newSelected = selectedMetrics.includes(metric.value)
                    ? selectedMetrics.filter((m) => m !== metric.value)
                    : [...selectedMetrics, metric.value];
                  onSelect(newSelected);
                }}
              >
                <Check
                  className={cn(
                    "mr-2 h-4 w-4",
                    selectedMetrics.includes(metric.value)
                      ? "opacity-100"
                      : "opacity-0"
                  )}
                />
                {metric.label}
              </CommandItem>
            ))}
          </CommandGroup>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
