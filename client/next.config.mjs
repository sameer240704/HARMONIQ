/** @type {import('next').NextConfig} */
const nextConfig = {
    images: {
        remotePatterns: [
            {
                protocol: "https",
                hostname: "images.unsplash.com",
            },
            {
                protocol: "https",
                hostname: "ui.aceternity.com"
            },
            {
                protocol: "https",
                hostname: "clunyfarm.co.za"
            },
            {
                protocol: "https",
                hostname: "www.devdiscourse.com",
            },
            {
                protocol: "https",
                hostname: "www.livemint.com",
            },
            {
                protocol: "https",
                hostname: "images.indianexpress.com"
            },
            {
                protocol: "https",
                hostname: "images.news18.com"
            },
            {
                protocol: "https",
                hostname: "www.businessinsider.in"
            },
            {
                protocol: "https",
                hostname: "blogs.sas.com"
            },
            {
                protocol: "https",
                hostname: "images.livemint.com"
            },
            {
                protocol: "https",
                hostname: "www.deccanherald.com"
            },
            {
                protocol: "https",
                hostname: "images.financialexpress.com"
            },
            {
                protocol: "https",
                hostname: "resize.indiatvnews.com"
            },
            {
                protocol: "https",
                hostname: "www.oneindia.com"
            }
        ],
    },
};

export default nextConfig;
