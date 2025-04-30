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

const banks = [
  "IDFC First Bank",
  "SBI",
  "Punjab National Bank",
  "HDFC Bank",
  "ICICI Bank",
  "Axis Bank",
  "Canara Bank",
  "Kotak Mahindra Bank",
];

export function BankSelector({ selectedBanks, onSelect, className }) {
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
          {selectedBanks.length > 0
            ? `${selectedBanks.length} bank(s) selected`
            : "Select banks..."}
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[300px] p-0">
        <Command>
          <CommandInput placeholder="Search banks..." />
          <CommandEmpty>No bank found.</CommandEmpty>
          <CommandGroup className="max-h-[300px] overflow-y-auto">
            {banks.map((bank) => (
              <CommandItem
                key={bank}
                onSelect={() => {
                  const newSelected = selectedBanks.includes(bank)
                    ? selectedBanks.filter((b) => b !== bank)
                    : [...selectedBanks, bank];
                  onSelect(newSelected);
                }}
              >
                <Check
                  className={cn(
                    "mr-2 h-4 w-4",
                    selectedBanks.includes(bank) ? "opacity-100" : "opacity-0"
                  )}
                />
                {bank}
              </CommandItem>
            ))}
          </CommandGroup>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
