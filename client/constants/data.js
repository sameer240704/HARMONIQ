import {
    TrendingUp,
    DollarSign,
    LineChart,
    Briefcase,
    Award,
    Globe,
} from "lucide-react";

export const bankColors = {
    "IDFC First Bank": "#6366f1",
    SBI: "#3b82f6",
    "Punjab National Bank": "#14b8a6",
    "HDFC Bank": "#f59e0b",
    "ICICI Bank": "#ef4444",
    "Axis Bank": "#8b5cf6",
    "Canara Bank": "#10b981",
    "Kotak Mahindra Bank": "#ec4899",
};

export const CATEGORIES = [
    {
        id: "all",
        label: "All News",
        icon: Globe,
        searchTerms: [
            "market trends",
            "stock analysis",
            "financial insights",
            "economic outlook",
            "investment strategies",
        ],
    },
    {
        id: "markets",
        label: "Markets",
        icon: TrendingUp,
        searchTerms: [
            "stock market",
            "market analysis",
            "market movements",
            "trading insights",
            "market volatility",
        ],
    },
    {
        id: "investments",
        label: "Investments",
        icon: DollarSign,
        searchTerms: [
            "investment strategies",
            "portfolio management",
            "asset allocation",
            "wealth creation",
            "investment returns",
        ],
    },
    {
        id: "economy",
        label: "Economy",
        icon: LineChart,
        searchTerms: [
            "economic indicators",
            "GDP growth",
            "inflation rates",
            "economic policy",
            "fiscal measures",
        ],
    },
    {
        id: "business",
        label: "Business",
        icon: Briefcase,
        searchTerms: [
            "corporate earnings",
            "business growth",
            "company performance",
            "mergers acquisitions",
            "business strategy",
        ],
    },
    {
        id: "insights",
        label: "Expert Insights",
        icon: Award,
        searchTerms: [
            "financial experts",
            "market analysis",
            "investment advice",
            "financial planning",
            "wealth management",
        ],
    },
];