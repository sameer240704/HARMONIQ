import * as React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUpRight, Clock, TrendingUp } from "lucide-react";

export function NewsFeed({ news, className }) {
  const groupedNews = news.reduce((acc, item) => {
    const date = new Date(item.publishedAt);
    const monthYear = date.toLocaleString("default", {
      month: "long",
      year: "numeric",
    });

    if (!acc[monthYear]) {
      acc[monthYear] = [];
    }
    acc[monthYear].push(item);
    return acc;
  }, {});

  return (
    <Card className={`${className} border-none shadow-lg`}>
      <CardContent className="p-0">
        {news.length === 0 ? (
          <div className="text-center text-muted-foreground py-12">
            <div className="mx-auto max-w-md">
              <p className="text-lg">No recent updates</p>
              <p className="text-sm mt-2">
                Check back later for the latest fintech news
              </p>
            </div>
          </div>
        ) : (
          <div className="">
            {Object.entries(groupedNews).map(([monthYear, items]) => (
              <div key={monthYear} className="py-4">
                <div className="px-6 mb-3 flex items-center gap-2">
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-gray-300 to-transparent dark:via-gray-700" />
                  <span className="text-xs font-medium text-gray-500 dark:text-gray-400 px-2">
                    {monthYear}
                  </span>
                  <div className="h-px flex-1 bg-gradient-to-r from-transparent via-gray-300 to-transparent dark:via-gray-700" />
                </div>
                <div className="space-y-4">
                  {items.map((item, index) => (
                    <div key={index} className="group px-6">
                      <a
                        href={item.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-start gap-4 hover:bg-gray-50 dark:hover:bg-gray-900/50 p-3 rounded-xl transition-all duration-200"
                      >
                        <div className="flex-shrink-0 mt-1">
                          <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-blue-100 to-indigo-100 dark:from-blue-900/30 dark:to-indigo-900/30 flex items-center justify-center">
                            <span className="text-blue-600 dark:text-blue-400 font-medium text-sm">
                              {item.source
                                .split(" ")
                                .map((w) => w[0])
                                .join("")
                                .substring(0, 2)
                                .toUpperCase()}
                            </span>
                          </div>
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <h3 className="font-semibold text-gray-900 dark:text-white group-hover:underline line-clamp-2">
                              {item.title}
                            </h3>
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
                            {item.description}
                          </p>
                          <div className="flex items-center gap-3 mt-2">
                            <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                              <Clock className="h-3 w-3" />
                              <span>
                                {new Date(item.publishedAt).toLocaleDateString(
                                  "en-US",
                                  {
                                    month: "short",
                                    day: "numeric",
                                  }
                                )}
                              </span>
                            </div>
                            <Badge
                              variant="outline"
                              className="text-xs px-2 py-0.5 rounded-full"
                            >
                              {item.source.replace(/\.com|\.net|www\./gi, "")}
                            </Badge>
                          </div>
                        </div>
                        <ArrowUpRight className="h-4 w-4 text-gray-400 group-hover:text-blue-600 dark:group-hover:text-blue-400 mt-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-all" />
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
