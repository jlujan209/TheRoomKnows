
export const metadata = {
  title: "The Room Knows",
  description: "The Smart Patient Exam Room",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
