import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Manual Answers - multimodal RAG",
  description: "Retrieve figures and tables from technical manuals with page citations",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
