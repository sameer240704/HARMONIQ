"use client";

import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  Loader2,
  Download,
  Save,
  BarChart3,
  LineChart,
  PieChart,
  TrendingUp,
  Award,
  ListFilter,
  ChevronRight,
  Calendar,
  Newspaper,
} from "lucide-react";

import { BankSelector } from "@/components/BankSelector";
import { MetricSelector } from "@/components/MetricSelector";
import { DateRangePicker } from "@/components/DateRangePicker";
import { NewsFeed } from "@/components/NewsFeed";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { toast } from "sonner";
import {
  GrowthRateChart,
  ITEfficiencyChart,
  MonteCarloForecastChart,
  RiskMetricsRadarChart,
  StockPriceChart,
} from "@/components/Charts";
import { Badge } from "@/components/ui/badge";
import Image from "next/image";
import { Logo } from "@/public/images";
import { useRouter } from "next/navigation";
import { API_BASE_URL } from "@/constants/utils";
import { bankColors } from "@/constants/data";

export default function Dashboard() {
  const [selectedBanks, setSelectedBanks] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [dateRange, setDateRange] = useState({
    start: new Date(new Date().setFullYear(new Date().getFullYear() - 1)),
    end: new Date(),
  });
  const [analysisResults, setAnalysisResults] = useState(null);
  const [bankFinancialData, setBankFinancialData] = useState({});
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [lastScrollY, setLastScrollY] = useState(0);
  const [activeBank, setActiveBank] = useState(null);
  const [riskMetrics, setRiskMetrics] = useState({});
  const [monteCarloForecast, setMonteCarloForecast] = useState(null);

  const MemoStockChart = React.memo(StockPriceChart);
  const MemoGrowthChart = React.memo(GrowthRateChart);
  const MemoEfficiencyChart = React.memo(ITEfficiencyChart);
  const MemoRiskMetricsRadarChart = React.memo(RiskMetricsRadarChart);
  const MemoMonteCarloForecastChart = React.memo(MonteCarloForecastChart);

  const router = useRouter();

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      if (currentScrollY > 100) {
        if (currentScrollY < lastScrollY) {
          setIsScrolled(true);
        } else if (currentScrollY > lastScrollY + 10) {
          setIsScrolled(false);
        }
      } else {
        setIsScrolled(false);
      }

      setLastScrollY(currentScrollY);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, [lastScrollY]);

  useEffect(() => {
    fetchNews();

    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => {
      window.removeEventListener("resize", checkMobile);
    };
  }, []);

  useEffect(() => {
    if (selectedBanks.length > 0) {
      fetchRiskMetrics(selectedBanks[0]);
      fetchMonteCarloForecast(selectedBanks[0]);
    }
  }, [selectedBanks]);

  const fetchNews = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/fetch-fintech-news`);
      setNews(response.data.news || []);
    } catch (error) {
      console.error("Error fetching news:", error);
      toast({
        title: "Error",
        description: "Failed to fetch news",
        variant: "destructive",
      });
    }
  };

  const runAnalysis = async () => {
    if (selectedBanks.length === 0) {
      toast({
        title: "Error",
        description: "Please select at least one bank",
        variant: "destructive",
      });
      return;
    }

    try {
      setLoading(true);
      setActiveBank(selectedBanks[0]);

      const resultsData = {
        stock_price: {},
        it_efficiency: {},
        digital_transactions: {},
        stats: {},
        comparison: [],
      };

      // Fetch financial data for each bank
      const financialDataPromises = selectedBanks.map((bankName) =>
        axios.get(
          `${API_BASE_URL}/financial-metrics?bank_name=${encodeURIComponent(
            bankName
          )}`
        )
      );

      const financialResults = await Promise.all(financialDataPromises);
      const bankFinancialMap = {};

      financialResults.forEach((result, index) => {
        const bankName = selectedBanks[index];
        const { metrics, timeline } = result.data;

        bankFinancialMap[bankName] = result.data;

        // Stock price data
        resultsData.stock_price[bankName] = timeline
          .filter((entry) => entry.date && entry.stock_price)
          .map((entry) => ({
            date: entry.date,
            value: Number(entry.stock_price) || 0,
          }));

        // IT Efficiency data
        resultsData.it_efficiency[bankName] = {
          value: Number(metrics.avg_it_spending) || 0,
          growth: Number(metrics.digital_growth) || 0,
          cloudAdoption: Number(metrics.cloud_adoption) || 0,
        };

        // Digital transactions
        resultsData.digital_transactions[bankName] = {
          mean: Number(metrics.avg_digital_transactions) || 0,
          growth_rate: Number(metrics.digital_growth) || 0,
        };

        // Stats data
        resultsData.stats[bankName] = {
          avg_it_news: Number(metrics.avg_it_news_count) || 0,
          avg_sentiment: Number(metrics.avg_it_sentiment) || 0,
          rnd_growth: Number(metrics.rnd_growth) || 0,
          cloud_adoption: Number(metrics.cloud_adoption) || 0,
          mobile_growth: Number(metrics.mobile_user_growth) || 0,
        };
      });

      setBankFinancialData(bankFinancialMap);

      // Fetch comparison data if multiple banks selected
      if (selectedBanks.length > 1) {
        try {
          const comparisonResponse = await axios.get(
            `${API_BASE_URL}/compare-banks?${selectedBanks
              .map((bank) => `bank_names=${encodeURIComponent(bank)}`)
              .join("&")}`
          );
          resultsData.comparison = comparisonResponse.data;
        } catch (error) {
          console.error("Error fetching comparison data:", error);
          resultsData.comparison = [];
        }
      }

      setAnalysisResults(resultsData);
      toast({
        title: "Success",
        description: "Analysis completed successfully",
      });
    } catch (error) {
      console.error("Error running analysis:", error);
      toast({
        title: "Error",
        description: error.response?.data?.detail || "Failed to run analysis",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const renderStatsCards = () => {
    if (!analysisResults) return null;

    const stats = [
      {
        title: "Banks Analyzed",
        value: selectedBanks.length,
        icon: <BarChart3 className="text-blue-500" />,
      },
      {
        title: "Avg Stock Price",
        value: `₹${(
          Object.values(analysisResults.stats).reduce(
            (sum, stats) => sum + (stats.avg_stock_price || 0),
            0
          ) / selectedBanks.length
        ).toFixed(2)}`,
        icon: <LineChart className="text-green-500" />,
      },
      {
        title: "Total R&D Spend",
        value: `₹${(
          Object.values(analysisResults.stats).reduce(
            (sum, stats) => sum + (stats.total_rnd || 0),
            0
          ) / 10000000
        ).toFixed(1)} Cr`,
        icon: <TrendingUp className="text-purple-500" />,
      },
    ];

    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {stats.map((stat, i) => (
          <Card key={i} className="shadow-sm hover:shadow-md transition-shadow">
            <CardContent className="flex items-center gap-4 p-4">
              <div className="h-10 w-10 rounded-lg bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                {stat.icon}
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {stat.title}
                </p>
                <p className="text-xl font-semibold">{stat.value}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  };

  const formatMetricName = (metric) => {
    return metric
      .replace(/_/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const fetchRiskMetrics = async (bankName) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/risk-metrics/${bankName}`
      );
      setRiskMetrics(response.data);
    } catch (error) {
      console.error("Error fetching risk metrics:", error);
      toast({
        title: "Error",
        description: "Failed to fetch risk metrics",
        variant: "destructive",
      });
    }
  };

  const fetchMonteCarloForecast = async (bankName, days = 30) => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/monte-carlo-forecast/${bankName}?days=${days}`
      );
      setMonteCarloForecast(response.data);
    } catch (error) {
      console.error("Error fetching Monte Carlo forecast:", error);
      toast({
        title: "Error",
        description: "Failed to fetch forecast data",
        variant: "destructive",
      });
    }
  };

  const renderBankMetricsTable = () => {
    if (!analysisResults?.stats) return null;

    return (
      <Card className="shadow-sm border-0">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-gray-50 dark:bg-gray-800">
              <TableRow>
                <TableHead>Bank</TableHead>
                <TableHead className="text-right">IT News</TableHead>
                <TableHead className="text-right">Sentiment</TableHead>
                <TableHead className="text-right">R&D Growth</TableHead>
                <TableHead className="text-right">Mobile Growth</TableHead>
                <TableHead className="text-right">Cloud Adoption</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(analysisResults.stats).map(([bank, stats]) => (
                <TableRow key={bank}>
                  <TableCell className="flex items-center gap-2">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: bankColors[bank] }}
                    />
                    {bank}
                  </TableCell>
                  <TableCell className="text-right">
                    {stats.avg_it_news.toFixed(1)}
                  </TableCell>
                  <TableCell className="text-right">
                    {stats.avg_sentiment > 0 ? "😊" : "😞"}{" "}
                    {stats.avg_sentiment.toFixed(2)}
                  </TableCell>
                  <TableCell className="text-right">
                    <span
                      className={
                        stats.rnd_growth >= 0
                          ? "text-green-600"
                          : "text-red-600"
                      }
                    >
                      {stats.rnd_growth.toFixed(1)}%
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    {stats.mobile_growth.toFixed(1)}%
                  </TableCell>
                  <TableCell className="text-right">
                    {(stats.cloud_adoption * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    );
  };

  const renderComparisonTable = () => {
    if (!analysisResults?.comparison?.length) return null;

    return (
      <Card className="shadow-sm border-0">
        <CardContent className="p-0">
          <Table>
            <TableHeader className="bg-gray-50 dark:bg-gray-800">
              <TableRow>
                <TableHead>Bank</TableHead>
                <TableHead className="text-right">IT Spending</TableHead>
                <TableHead className="text-right">
                  Digital Transactions
                </TableHead>
                <TableHead className="text-right">Cloud Score</TableHead>
                <TableHead className="text-right">Fintech Partners</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {analysisResults.comparison.map((bank) => (
                <TableRow key={bank.bank_name}>
                  <TableCell className="flex items-center gap-2">
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: bankColors[bank.bank_name] }}
                    />
                    {bank.bank_name}
                  </TableCell>
                  <TableCell className="text-right">
                    ₹{(bank.avg_it_spending / 1000000).toFixed(1)}M
                  </TableCell>
                  <TableCell className="text-right">
                    {(bank.total_digital / 1000).toFixed(1)}K
                  </TableCell>
                  <TableCell className="text-right">
                    {(bank.avg_cloud_adoption * 100).toFixed(1)}%
                  </TableCell>
                  <TableCell className="text-right">
                    {bank.total_fintech_partnerships}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    );
  };

  const RiskMetricsCard = () => {
    if (!riskMetrics || Object.keys(riskMetrics).length === 0) return null;

    return (
      <Card className="shadow-sm border-0 h-full">
        <CardHeader className="bg-gray-50 dark:bg-gray-800 py-4 px-6 border-b">
          <CardTitle className="text-sm font-medium flex items-center">
            <BarChart3 className="h-4 w-4 text-blue-500 mr-2" />
            Risk Metrics Profile
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 h-[400px]">
          <MemoRiskMetricsRadarChart metrics={riskMetrics} />
        </CardContent>
      </Card>
    );
  };

  const MonteCarloForecastCard = () => {
    if (!monteCarloForecast) return null;

    return (
      <Card className="shadow-sm border-0 h-full">
        <CardHeader className="bg-gray-50 dark:bg-gray-800 py-4 px-6 border-b">
          <CardTitle className="text-sm font-medium flex items-center">
            <TrendingUp className="h-4 w-4 text-green-500 mr-2" />
            Monte Carlo Forecast ({monteCarloForecast.forecast_days} days)
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 h-[400px]">
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-500">Current Price</p>
              <p className="text-lg font-semibold">
                ₹{monteCarloForecast.current_price.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Expected Return</p>
              <p className="text-lg font-semibold">
                {(monteCarloForecast.expected_return * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Volatility</p>
              <p className="text-lg font-semibold">
                {(monteCarloForecast.volatility * 100).toFixed(2)}%
              </p>
            </div>
          </div>
          <MemoMonteCarloForecastChart forecast={monteCarloForecast} />
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 text-black dark:bg-gray-900 pb-12">
      <div
        className={`
        fixed top-4 left-1/2 -translate-x-1/2 z-50 
        w-[calc(100%-2rem)] max-w-4xl mx-auto
        transition-all duration-300 ease-in-out
        ${
          isScrolled ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-20"
        }
      `}
      >
        <div
          className="
          bg-white dark:bg-gray-800 
          rounded-full shadow-xl 
          border border-gray-200 dark:border-gray-700
          px-6 py-3
          flex items-center justify-between
        "
        >
          <div className="flex items-center gap-2">
            <div className="flex justify-center items-center gap-3">
              <Image src={Logo} alt="Logo" className="h-8 w-auto rounded-sm" />
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
                HARMONIQ
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="rounded-full">
              <Download className="h-4 w-4" />
            </Button>
            <Button size="sm" className="rounded-full">
              <Save className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div
        className={`pt-4 mb-4 transition-opacity ${
          isScrolled ? "opacity-0" : "opacity-100"
        }`}
      >
        <div className="container mx-auto max-w-7xl px-5 mb-5">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div className="flex justify-center items-center gap-3">
              <Image src={Logo} alt="Logo" className="h-10 w-auto rounded-md" />
              <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
                HARMONIQ
              </h1>
            </div>
            <div className="flex items-center gap-2 w-full md:w-auto">
              <Button variant="ghost" className="cursor-pointer">
                <Download className="h-4 w-4" />
                Export
              </Button>
              <Button className="w-full md:w-auto cursor-pointer">
                <Save className="h-4 w-4" />
                Save Analysis
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto max-w-7xl px-4">
        <div className="flex flex-col gap-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 sticky top-4 h-fit">
              <Card className="lg:col-span-1 shadow-md hover:shadow-lg transition-shadow">
                <CardHeader className="">
                  <CardTitle className="flex items-center text-lg">
                    <ListFilter className="mr-2 h-5 w-5" />
                    Analysis Parameters
                  </CardTitle>
                  <CardDescription>
                    Select banks, metrics and date range to analyze
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                      Banks
                    </label>
                    <BankSelector
                      selectedBanks={selectedBanks}
                      onSelect={setSelectedBanks}
                    />
                    {selectedBanks.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {selectedBanks.map((bank) => (
                          <Badge
                            key={bank}
                            style={{ backgroundColor: bankColors[bank] }}
                            className={`text-xs py-0.5 text-white`}
                          >
                            {bank}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                      Metrics
                    </label>
                    <MetricSelector
                      selectedMetrics={selectedMetrics}
                      onSelect={setSelectedMetrics}
                    />
                    {selectedMetrics.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 mt-2">
                        {selectedMetrics.map((metric) => (
                          <Badge
                            key={metric}
                            className="text-xs py-0.5 bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-100"
                          >
                            {formatMetricName(metric)}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                      Date Range
                    </label>
                    <DateRangePicker
                      dateRange={dateRange}
                      onSelect={setDateRange}
                    />
                  </div>

                  <Button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="w-full cursor-pointer"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <TrendingUp className="mr-2 h-4 w-4" />
                        Run Analysis
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>
            </div>

            <div className="lg:col-span-2">
              <Card className="h-fit lg:col-span-3 shadow-md hover:shadow-lg transition-shadow">
                <CardHeader className="">
                  <CardTitle className="flex items-center text-lg">
                    <LineChart className="mr-2 h-5 w-5" />
                    Analysis Results
                  </CardTitle>
                  <CardDescription>
                    View detailed analysis of selected banking metrics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {analysisResults ? (
                    <>
                      {renderStatsCards()}

                      <Tabs defaultValue="overview" className="">
                        <TabsList className="grid w-full grid-cols-4 rounded-lg">
                          <TabsTrigger
                            value="overview"
                            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700"
                          >
                            Overview
                          </TabsTrigger>
                          <TabsTrigger
                            value="metrics"
                            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700"
                          >
                            Metrics
                          </TabsTrigger>
                          <TabsTrigger
                            value="comparison"
                            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700"
                          >
                            Comparison
                          </TabsTrigger>
                          <TabsTrigger
                            value="risk"
                            className="data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700"
                          >
                            Risk
                          </TabsTrigger>
                        </TabsList>

                        <TabsContent value="overview" className="pt-6">
                          <div className="grid grid-cols-1 gap-6">
                            {analysisResults.stock_price && (
                              <Card className="shadow-sm border-0 overflow-hidden">
                                <CardHeader className="bg-gray-50 dark:bg-gray-800 py-4 px-6 border-b">
                                  <CardTitle className="text-sm font-medium flex items-center">
                                    <BarChart3 className="h-4 w-4 text-blue-500 mr-2" />
                                    Stock Price Analysis
                                  </CardTitle>
                                </CardHeader>
                                <CardContent className="p-6">
                                  <MemoStockChart
                                    data={analysisResults.stock_price}
                                  />
                                </CardContent>
                              </Card>
                            )}

                            {analysisResults.stock_price && (
                              <Card className="shadow-sm border-0 overflow-hidden">
                                <CardHeader className="bg-gray-50 dark:bg-gray-800 py-4 px-6 border-b">
                                  <CardTitle className="text-sm font-medium flex items-center">
                                    <TrendingUp className="h-4 w-4 text-green-500 mr-2" />
                                    Growth Rate Analysis
                                  </CardTitle>
                                </CardHeader>
                                <CardContent className="p-6">
                                  <MemoGrowthChart
                                    data={analysisResults.stock_price}
                                    title="Digital Transactions Growth"
                                  />
                                </CardContent>
                              </Card>
                            )}

                            {analysisResults.it_efficiency && (
                              <Card className="shadow-sm border-0 overflow-hidden">
                                <CardHeader className="bg-gray-50 dark:bg-gray-800 py-4 px-6 border-b">
                                  <CardTitle className="text-sm font-medium flex items-center">
                                    <Award className="h-4 w-4 text-purple-500 mr-2" />
                                    IT Efficiency Analysis
                                  </CardTitle>
                                </CardHeader>
                                <CardContent className="p-6">
                                  <ITEfficiencyChart
                                    data={analysisResults.it_efficiency}
                                    metrics={[
                                      "value",
                                      "growth",
                                      "cloudAdoption",
                                    ]}
                                  />
                                </CardContent>
                              </Card>
                            )}
                          </div>
                        </TabsContent>

                        <TabsContent value="metrics" className="pt-6">
                          {renderBankMetricsTable()}
                        </TabsContent>

                        <TabsContent value="comparison" className="pt-6">
                          <div className="grid grid-cols-1 gap-6 mb-6">
                            {analysisResults.it_efficiency && (
                              <Card className="shadow-sm border-0 overflow-hidden">
                                <CardHeader className="bg-gray-50 dark:bg-gray-800 py-4 px-6 border-b">
                                  <CardTitle className="text-sm font-medium flex items-center">
                                    <Award className="h-4 w-4 text-purple-500 mr-2" />
                                    IT Efficiency Comparison
                                  </CardTitle>
                                </CardHeader>
                                <CardContent className="p-6">
                                  <MemoEfficiencyChart
                                    data={analysisResults.it_efficiency}
                                    metrics={["cloudAdoption"]}
                                  />
                                </CardContent>
                              </Card>
                            )}
                          </div>

                          {renderComparisonTable()}
                        </TabsContent>

                        <TabsContent value="risk" className="pt-6">
                          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            <RiskMetricsCard metrics={riskMetrics} />
                            <MonteCarloForecastCard
                              forecast={monteCarloForecast}
                            />
                          </div>
                        </TabsContent>
                      </Tabs>
                    </>
                  ) : (
                    <div className="h-96 flex flex-col items-center justify-center gap-4 text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                      <div className="h-20 w-20 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                        <LineChart className="h-10 w-10" />
                      </div>
                      <div className="text-center">
                        <p className="text-lg font-medium">
                          No analysis results yet
                        </p>
                        <p className="text-sm mt-1 max-w-sm">
                          Select banks, metrics, and a date range, then click
                          "Run Analysis" to view insights
                        </p>
                      </div>
                      <Button
                        variant="primary"
                        onClick={runAnalysis}
                        disabled={selectedBanks.length === 0 || loading}
                        className="mt-2 text-white"
                      >
                        {loading ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <TrendingUp className="h-4 w-4" />
                            Run Analysis
                          </>
                        )}
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>

          <div className="mt-4">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold flex items-center">
                <Newspaper className="h-5 w-auto mr-2" />
                Latest Banking & Fintech News
              </h2>
              <Button
                variant="outline"
                size="sm"
                className="text-xs flex items-center gap-1 cursor-pointer"
                onClick={() => router.push("/news")}
              >
                View All <ChevronRight className="h-3 w-3" />
              </Button>
            </div>
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md">
              <NewsFeed news={news} className="rounded-lg" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
