import React from "react";
import dynamic from "next/dynamic";
import { bankColors } from "@/constants/data";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export const StockPriceChart = ({ data }) => {
  if (!data || Object.keys(data).length === 0) return null;

  const banks = Object.keys(data);
  const dates = data[banks[0]].map((item) => item.date);

  const chartData = banks.map((bank) => ({
    x: dates,
    y: data[bank].map((item) => item.value),
    type: "line",
    name: bank,
    line: { color: bankColors[bank] },
    hovertemplate: `%{y:.2f}<extra>${bank}</extra>`,
  }));

  return (
    <Plot
      data={chartData}
      layout={{
        title: "Stock Price Trends",
        xaxis: { title: "Date" },
        yaxis: { title: "Price (₹)" },
        hovermode: "x unified",
        showlegend: banks.length > 1,
      }}
      config={{ responsive: true }}
    />
  );
};

export const GrowthRateChart = ({ data, title = "Growth Rate" }) => {
  if (!data || Object.keys(data).length === 0) return null;

  const banks = Object.keys(data);

  const calculateGrowth = (values) => {
    const first = values[0]?.value;
    const last = values[values.length - 1]?.value;

    if (!first || !last) return 0;
    return ((last - first) / first) * 100; // Percentage change
  };

  const chartData = [
    {
      x: banks,
      y: banks.map((bank) => calculateGrowth(data[bank])),
      type: "bar",
      marker: {
        color: banks.map((bank) => {
          const growth = calculateGrowth(data[bank]);
          return growth >= 0 ? "#4CAF50" : "#F44336";
        }),
      },
      hovertemplate: "%{y:.2f}%<extra>%{x}</extra>",
      text: banks.map((bank) => calculateGrowth(data[bank]).toFixed(2) + "%"),
      textposition: "auto",
    },
  ];

  // Calculate reasonable y-axis range
  const allGrowthRates = banks.map((bank) => calculateGrowth(data[bank]));
  const maxRate = Math.max(...allGrowthRates.map(Math.abs));
  const yRange = [-maxRate * 1.1, maxRate * 1.1]; // Symmetrical range

  return (
    <Plot
      data={chartData}
      layout={{
        title: `${title} (Period Change)`,
        xaxis: { title: "Bank" },
        yaxis: {
          title: "Price Change (%)",
          range: yRange,
          ticksuffix: "%",
        },
        hovermode: "closest",
        showlegend: false,
        margin: { t: 40, b: 80, l: 60, r: 40 },
      }}
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }}
      style={{ width: "100%", height: "400px" }}
      useResizeHandler={true}
    />
  );
};

export const ITEfficiencyChart = ({
  data,
  metrics = ["value", "growth", "cloudAdoption"],
  type = "bar",
  title = "IT Efficiency Metrics",
}) => {
  if (!data || Object.keys(data).length === 0) return null;

  const banks = Object.keys(data);
  const metricNames = {
    value: "IT Spending (₹ Cr)",
    growth: "Growth Rate (%)",
    cloudAdoption: "Cloud Adoption (%)",
  };

  // Convert values to appropriate formats
  const chartData = metrics.map((metric) => {
    const isCloud = metric === "cloudAdoption";
    const isGrowth = metric === "growth";

    return {
      x: banks,
      y: banks.map((bank) => {
        const rawValue = data[bank][metric];
        if (isCloud) return rawValue * 100; // Convert to percentage
        if (isGrowth) return rawValue * 100; // Convert growth to percentage
        return rawValue / 10000000; // Convert IT Spending to Crores
      }),
      name: metricNames[metric],
      type: type === "pie" ? "pie" : "bar",
      marker: {
        color: banks.map((bank) => bankColors[bank]),
      },
      hovertemplate:
        isCloud || isGrowth
          ? `<b>%{x}</b><br>${metricNames[metric]}: %{y:.2f}%`
          : `<b>%{x}</b><br>${metricNames[metric]}: ₹%{y:.2f} Cr`,
      yaxis: isCloud ? "y2" : "y",
      visible: true,
    };
  });

  // Axis configuration
  const axisConfig = {
    xaxis: { title: "Bank" },
    yaxis: {
      title: "IT Spending / Growth Rate",
      tickprefix: metricNames.growth.includes("%") ? "" : "₹",
      ticksuffix: metricNames.growth.includes("%") ? "%" : "",
      showgrid: true,
    },
    yaxis2: {
      title: "Cloud Adoption (%)",
      overlaying: "y",
      side: "right",
      range: [0, 100],
      ticksuffix: "%",
      showgrid: false,
    },
  };

  return (
    <Plot
      data={chartData}
      layout={{
        ...axisConfig,
        title,
        barmode: metrics.length > 1 ? "group" : "stack",
        showlegend: metrics.length > 1,
        hovermode: "x unified",
        margin: { t: 40, b: 80, l: 75, r: 75 },
      }}
      config={{
        responsive: true,
        displayModeBar: true,
        displaylogo: false,
      }}
      style={{ width: "100%", height: "500px" }}
      useResizeHandler={true}
    />
  );
};

export const MonteCarloForecastChart = ({ forecast }) => {
  if (!forecast) return null;

  // Prepare data for Plotly
  const traces = forecast.paths.slice(0, 100).map((path, i) => ({
    y: path,
    type: "scatter",
    mode: "lines",
    line: { width: 1, color: "#3b82f6" },
    opacity: 0.4,
    name: `Path ${i + 1}`,
    hoverinfo: "none",
  }));

  // Add percentiles
  const percentiles = [
    {
      y: Array(forecast.paths[0].length).fill(forecast.percentiles["5th"]),
      type: "scatter",
      mode: "lines",
      line: { width: 2, color: "#ef4444" },
      name: "5th Percentile",
    },
    {
      y: Array(forecast.paths[0].length).fill(forecast.percentiles["50th"]),
      type: "scatter",
      mode: "lines",
      line: { width: 2, color: "#10b981" },
      name: "Median (50th)",
    },
    {
      y: Array(forecast.paths[0].length).fill(forecast.percentiles["95th"]),
      type: "scatter",
      mode: "lines",
      line: { width: 2, color: "#3b82f6" },
      name: "95th Percentile",
    },
  ];

  const data = [...traces, ...percentiles];

  return (
    <Plot
      data={data}
      layout={{
        title: `${forecast.forecast_days}-Day Price Forecast`,
        xaxis: { title: "Days Ahead" },
        yaxis: { title: "Price (₹)" },
        showlegend: true,
        hovermode: "closest",
        margin: { t: 40, b: 40, l: 60, r: 40 },
        legend: { orientation: "h", y: -0.2 },
      }}
      config={{
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
      }}
      className="w-full h-full"
    />
  );
};

export const RiskMetricsRadarChart = ({ metrics }) => {
  if (!metrics || Object.keys(metrics).length === 0) return null;

  const riskData = [
    {
      type: "scatterpolar",
      r: [
        metrics.daily_volatility * 100,
        metrics.annualized_volatility * 100,
        Math.abs(metrics.value_at_risk_95 * 100),
        Math.abs(metrics.conditional_var_95 * 100),
        Math.abs(metrics.returns_skewness),
        Math.abs(metrics.returns_kurtosis),
      ],
      theta: [
        "Daily Volatility",
        "Annual Volatility",
        "VaR (95%)",
        "CVaR (95%)",
        "Skewness",
        "Kurtosis",
      ],
      fill: "toself",
      name: "Risk Profile",
      marker: { color: "#6366f1" },
    },
  ];

  return (
    <Plot
      data={riskData}
      layout={{
        title: "Risk Metrics Profile",
        polar: {
          radialaxis: {
            visible: true,
            range: [
              0,
              Math.max(
                metrics.annualized_volatility * 100,
                Math.abs(metrics.value_at_risk_95 * 100),
                3
              ),
            ],
          },
        },
        showlegend: true,
        margin: { t: 40, b: 40, l: 60, r: 60 },
      }}
      config={{
        responsive: true,
        displayModeBar: true,
      }}
      className="w-full h-full"
    />
  );
};
