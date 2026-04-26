import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Succession-Planning Graph",
  description: "Spectral embeddings + skills + structural proximity for succession recommendations",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
