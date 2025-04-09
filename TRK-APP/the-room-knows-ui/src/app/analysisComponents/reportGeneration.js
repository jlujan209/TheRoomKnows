"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";

const ReportGeneration = ({ patient_name }) => {
  const router = useRouter();
  const [pdfUrl, setPdfUrl] = useState(null);

  useEffect(() => {
    if (patient_name) {
      // Example PDF endpoint
      setPdfUrl(`/api/visit-report?patient_name=${encodeURIComponent(patient_name)}`);
    }
  }, [patient_name]);

  return (
    <div className="container py-4 text-center">
      <h2 className="mb-4">Final Visit Report for {patient_name}</h2>

      {pdfUrl ? (
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
        <div className="alert alert-info">Loading report...</div>
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
