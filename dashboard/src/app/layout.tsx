import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import Nav from "@/components/Nav";
import BottomNav from "@/components/BottomNav";
import DataSourceBanner from "@/components/DataSourceBanner";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "900"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "OptiNet — Trading Control",
  description:
    "Live-testing control room for the NIFTY futures router and IntradayNet equity systems.",
  icons: {
    icon: [{ url: "/favicon.svg", type: "image/svg+xml" }],
    shortcut: "/favicon.svg",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#030303",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className={`${inter.className} relative min-h-screen`}>
        {/* ambient top glow */}
        <div className="glow-violet pointer-events-none absolute inset-x-0 top-0 -z-10 h-[420px]" />
        <Nav />
        <main className="mx-auto w-full max-w-[1180px] px-4 pb-28 pt-8 sm:px-6 md:pb-24">
          <DataSourceBanner />
          {children}
        </main>
        <BottomNav />
      </body>
    </html>
  );
}
