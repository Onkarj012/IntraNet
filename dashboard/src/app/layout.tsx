import type { Metadata, Viewport } from "next";
import Nav from "@/components/Nav";
import BottomNav from "@/components/BottomNav";
import "./globals.css";

export const metadata: Metadata = {
  title: "OptiNet — Trading Control",
  description:
    "Live-testing control room for the NIFTY futures router and IntradayNet equity systems.",
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
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="relative min-h-screen">
        {/* ambient top glow */}
        <div className="glow-violet pointer-events-none absolute inset-x-0 top-0 -z-10 h-[420px]" />
        <Nav />
        <main className="mx-auto w-full max-w-[1180px] px-4 pb-28 pt-8 sm:px-6 md:pb-24">
          {children}
        </main>
        <BottomNav />
      </body>
    </html>
  );
}
