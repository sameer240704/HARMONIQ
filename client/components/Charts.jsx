import React from "react";
import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export const StockPriceChart = ({ data }) => {
  if (!data) return null;

  const banks = Object.keys(data);
  const metrics = ["mean", "median", "min", "max"];

  const chartData = metrics.map((metric) => ({
    x: banks,
    y: banks.map((bank) => data[bank][metric]),
    type: "bar",
    name: metric,
  }));

  return (
    <Plot
      data={chartData}
      layout={{
        title: "Stock Price Metrics",
        barmode: "group",
        xaxis: { title: "Bank" },
        yaxis: { title: "Price" },
      }}
    />
  );
};

export const GrowthRateChart = ({ data }) => {
  if (!data) return null;

  const banks = Object.keys(data);

  const chartData = [
    {
      x: banks,
      y: banks.map((bank) => data[bank].growth_rate),
      type: "bar",
      marker: {
        color: banks.map((bank) =>
          data[bank].growth_rate > 0 ? "green" : "red"
        ),
      },
    },
  ];

  return (
    <Plot
      data={chartData}
      layout={{
        title: "Stock Price Growth Rate",
        xaxis: { title: "Bank" },
        yaxis: { title: "Growth Rate" },
      }}
    />
  );
};

export const ITEfficiencyChart = ({ data }) => {
  if (!data) return null;

  const banks = Object.keys(data);

  const chartData = [
    {
      x: banks,
      y: banks.map((bank) => data[bank]),
      type: "bar",
    },
  ];

  return (
    <Plot
      data={chartData}
      layout={{
        title: "IT Efficiency",
        xaxis: { title: "Bank" },
        yaxis: { title: "Efficiency Score" },
      }}
    />
  );
};
