"use client";
import { useRouter } from "next/navigation";
import Image from "next/image";

export default function Home() {
  const router = useRouter();

  return (
    <div className="container d-flex flex-column align-items-center text-center py-5">
      <h1 className="display-4 fw-bold text-primary mb-3">The Room Knows</h1>
      <div className="mb-4">
        <Image
          src="/PatientExamRoom.jpg"
          width={250}
          height={250}
          alt="Patient Exam Room"
          className="rounded shadow"
        />
      </div>

      <div className="col-lg-8">
        <p className="lead text-muted">
          The growing demand for primary healthcare has led to shorter appointment times,
          reducing meaningful doctor-patient interactions. Current electronic health record 
          systems require extensive manual data entry, taking time away from patient care.
        </p>
        
        <p className="text-muted">
          Our system integrates audio, facial, and motion analysis to automate observation 
          and documentation. It captures verbal and non-verbal cues, analyzes speech and 
          emotions, detects facial abnormalities, and evaluates patient movement. This 
          information is compiled into a real-time diagnostic report, assisting physicians 
          in improving clinical efficiency and diagnostic accuracy.
        </p>
      </div>

      <button
        type="button"
        className="btn btn-lg text-white px-4 mt-3"
        style={{ backgroundColor: "#AB0520" }}
        onClick={() => router.push("/home")}
      >
        Get Started
      </button>
    </div>
  );
}
