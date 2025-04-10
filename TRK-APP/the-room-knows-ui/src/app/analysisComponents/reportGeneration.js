"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

const ReportGeneration = ({ patient_name }) => {
  const router = useRouter();
  const [pdfUrl, setPdfUrl] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const generateReport = async () => {
      try {
        const res = await fetch(`http://localhost:5000/generate-report/${encodeURIComponent(patient_name)}`);
        const data = await res.json();

        if (res.ok && data.filepath) {
          setPdfUrl(data.filepath);
        } else {
          console.error("Failed to generate report", data);
        }
      } catch (err) {
        console.error("Error calling report endpoint:", err);
      } finally {
        setLoading(false);
      }
    };

    if (patient_name) {
      generateReport();
    }
  }, [patient_name]);

  return (
    <div className="container py-4 text-center">
      <h2 className="mb-4">Final Visit Report for {patient_name}</h2>

      {loading ? (
        <div className="alert alert-info">Generating report...</div>
      ) : pdfUrl ? (
        <div className="mb-4 d-flex justify-content-center">
          <iframe
            src={pdfUrl}
            width="100%"
            height="600px"
            style={{ border: "1px solid #ccc", borderRadius: "8px", maxWidth: "900px" }}
            title="Visit Report PDF"
          />
        </div>
      ) : (
        <div className="alert alert-danger">Failed to load report.</div>
      )}

      <button
        onClick={() => router.push("/home")}
        className="btn btn-primary"
      >
        Back to Dashboard
      </button>
    </div>
  );
};

export default ReportGeneration;
