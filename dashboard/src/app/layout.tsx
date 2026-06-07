import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import Nav from "@/components/Nav";
import BottomNav from "@/components/BottomNav";
import DataSourceBanner from "@/components/DataSourceBanner";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "OptiNet",
  description: "Live trading control room — NIFTY futures router and IntradayNet equity.",
  icons: { icon: [{ url: "/favicon.svg", type: "image/svg+xml" }] },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#0a0a0a",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link
          href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap"
          rel="stylesheet"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className={inter.className}>
        <Nav />
        <main className="mx-auto max-w-[1200px] px-5 pb-24 pt-10 sm:pb-20">
          <DataSourceBanner />
          {children}
        </main>
        <BottomNav />
      </body>
    </html>
  );
}
