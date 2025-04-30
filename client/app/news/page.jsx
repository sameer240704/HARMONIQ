"use client";

import React, { useState, useEffect } from "react";
import {
  Clock,
  ExternalLink,
  Search,
  Loader2,
  LayoutGrid,
  AlignJustify,
  ChevronLeft,
} from "lucide-react";
import Image from "next/image";
import { CATEGORIES } from "@/constants/data";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

const NewsPage = () => {
  const [selectedCategory, setSelectedCategory] = useState(CATEGORIES[0]);
  const [searchQuery, setSearchQuery] = useState("");
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [viewStyle, setViewStyle] = useState("grid");

  const router = useRouter();

  const fetchNews = async (category) => {
    setLoading(true);
    setError("");
    try {
      const searchTerms = category.searchTerms
        .map((term) => `"${term}"`)
        .join(" OR ");

      const contextTerms = "finance OR markets OR economy OR investment";

      const finalQuery = encodeURIComponent(
        `(${searchTerms}) AND (${contextTerms})`
      );

      const response = await fetch(
        `https://gnews.io/api/v4/search?q=${finalQuery}&lang=en&country=in&max=9&sortby=relevance&apikey=${process.env.NEXT_PUBLIC_GNEWS_API_KEY}`
      );

      const data = await response.json();

      if (data.errors) {
        throw new Error(data.errors[0]);
      }

      const validArticles = (data.articles || []).filter(
        (article) =>
          article.title &&
          article.description &&
          article.url &&
          article.publishedAt
      );

      const uniqueArticles = Array.from(
        new Map(validArticles.map((article) => [article.url, article])).values()
      );

      setNews(uniqueArticles);
    } catch (err) {
      setError("Failed to fetch financial news. Please try again later.");
      console.error("Error fetching news:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNews(selectedCategory);
  }, [selectedCategory]);

  const filteredNews = news.filter(
    (article) =>
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    );

    if (diffInHours < 1) {
      return "Just now";
    } else if (diffInHours < 24) {
      return `${diffInHours} hours ago`;
    } else {
      return `${Math.floor(diffInHours / 24)} days ago`;
    }
  };

  const isHotNews = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInHours = Math.floor(
      (now.getTime() - date.getTime()) / (1000 * 60 * 60)
    );
    return diffInHours < 12;
  };

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900">
      <div className="container mx-auto max-w-7xl px-4 py-4">
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            size="sm"
            className="flex items-center gap-1.5 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-full border border-gray-200 shadow-sm hover:shadow transition-all cursor-pointer"
            onClick={() => router.push("/")}
          >
            <ChevronLeft className="h-4 w-4" />
            <span>Go to Home</span>
          </Button>
        </div>

        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-4">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search financial news..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:ring-2 focus:ring-black dark:focus:ring-white focus:border-transparent shadow-sm"
            />
          </div>

          <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
            <button
              onClick={() => setViewStyle("grid")}
              className={`p-2 rounded-md cursor-pointer ${
                viewStyle === "grid"
                  ? "bg-white dark:bg-gray-700 shadow-sm"
                  : "text-gray-600 dark:text-gray-400"
              }`}
              aria-label="Grid view"
            >
              <LayoutGrid className="h-5 w-5" />
            </button>
            <button
              onClick={() => setViewStyle("list")}
              className={`p-2 rounded-md cursor-pointer ${
                viewStyle === "list"
                  ? "bg-white dark:bg-gray-700 shadow-sm"
                  : "text-gray-600 dark:text-gray-400"
              }`}
              aria-label="List view"
            >
              <AlignJustify className="h-5 w-5" />
            </button>
          </div>
        </div>

        <div className="flex items-center space-x-2 overflow-x-auto pb-4 mb-8 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600">
          {CATEGORIES.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all duration-200 flex items-center whitespace-nowrap cursor-pointer ${
                selectedCategory.id === category.id
                  ? "bg-black text-white dark:bg-white dark:text-black shadow-md"
                  : "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
              }`}
            >
              <category.icon className="h-4 w-4 mr-2" />
              {category.label}
            </button>
          ))}
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20">
            <div className="flex flex-col items-center">
              <Loader2 className="h-8 w-8 text-black dark:text-white animate-spin mb-4" />
              <span className="text-gray-600 dark:text-gray-400">
                Loading financial insights...
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 text-center">
            <p className="text-red-600 dark:text-red-400">{error}</p>
            <button
              onClick={() => fetchNews(selectedCategory)}
              className="mt-4 px-4 py-2 bg-black text-white dark:bg-white dark:text-black rounded-lg hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && (
          <>
            {viewStyle === "grid" ? (
              <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {filteredNews.map((article, index) => (
                  <article
                    key={index}
                    className="bg-white dark:bg-gray-800 rounded-lg shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden border border-gray-100 dark:border-gray-700 group"
                  >
                    <div className="relative h-48 w-full overflow-hidden">
                      <Image
                        src={
                          article.image ||
                          "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop"
                        }
                        alt={article.title}
                        fill
                        className="object-cover group-hover:scale-105 transition-transform duration-300"
                        onError={(e) => {
                          e.target.src =
                            "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop";
                        }}
                      />
                      {isHotNews(article.publishedAt) && (
                        <div className="absolute top-2 right-2 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full flex items-center">
                          <span className="animate-pulse mr-1">●</span> BREAKING
                        </div>
                      )}
                    </div>
                    <div className="p-5">
                      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-3">
                        <div className="flex items-center">
                          <Clock className="h-3 w-3 mr-1" />
                          {formatDate(article.publishedAt)}
                        </div>
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full">
                          {article.source?.name || "Unknown"}
                        </span>
                      </div>
                      <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-2 line-clamp-2">
                        {article.title}
                      </h2>
                      <p className="text-gray-600 dark:text-gray-300 mb-4 line-clamp-3 text-sm">
                        {article.description}
                      </p>
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-sm font-medium text-black dark:text-white hover:underline"
                      >
                        Read Analysis <ExternalLink className="h-4 w-4 ml-1" />
                      </a>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                {filteredNews.map((article, index) => (
                  <article
                    key={index}
                    className="bg-white dark:bg-gray-800 rounded-lg shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden border border-gray-100 dark:border-gray-700 flex flex-col sm:flex-row"
                  >
                    <div className="sm:w-1/3 h-48 sm:h-auto relative">
                      <Image
                        src={
                          article.image ||
                          "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop"
                        }
                        alt={article.title}
                        fill
                        className="object-cover"
                        onError={(e) => {
                          e.target.src =
                            "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop";
                        }}
                      />
                      {isHotNews(article.publishedAt) && (
                        <div className="absolute top-2 left-2 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full flex items-center">
                          <span className="animate-pulse mr-1">●</span> BREAKING
                        </div>
                      )}
                    </div>
                    <div className="sm:w-2/3 p-5">
                      <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400 mb-2">
                        <div className="flex items-center">
                          <Clock className="h-3 w-3 mr-1" />
                          {formatDate(article.publishedAt)}
                        </div>
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded-full">
                          {article.source?.name || "Unknown"}
                        </span>
                      </div>
                      <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                        {article.title}
                      </h2>
                      <p className="text-gray-600 dark:text-gray-300 mb-3 line-clamp-2 text-sm">
                        {article.description}
                      </p>
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-sm font-medium text-black dark:text-white hover:underline"
                      >
                        Read Analysis <ExternalLink className="h-4 w-4 ml-1" />
                      </a>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </>
        )}

        {!loading && !error && filteredNews.length === 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700 p-8 text-center">
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              No financial news found matching your criteria
            </p>
            <button
              onClick={() => {
                setSearchQuery("");
                setSelectedCategory(CATEGORIES[0]);
              }}
              className="px-4 py-2 bg-black text-white dark:bg-white dark:text-black rounded-lg hover:bg-gray-800 dark:hover:bg-gray-200 transition-colors"
            >
              Reset Filters
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default NewsPage;
