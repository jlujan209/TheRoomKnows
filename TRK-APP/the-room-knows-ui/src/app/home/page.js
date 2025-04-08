"use client";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import withAuth from "../hoc/withAuth";

const api_key = process.env.NEXT_PUBLIC_API_KEY;

function Home() {
  const router = useRouter();
  const [patientsList, setPatientsList] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const handleLogout = () => {
    localStorage.removeItem("authToken"); 
    router.push("/login"); 
  };

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await fetch("http://localhost:5000/patients/all", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "API-Key": api_key,
          },
        });
        if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json();
        setPatientsList(data.patients);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPatients();
  }, []);

  return (
    <div className="container mt-5">
      <div className="d-flex justify-content-end mb-3">
        <button
          onClick={handleLogout}
          className="btn btn-danger"
          style={{ borderRadius: "8px" }}
        >
          Logout
        </button>
      </div>

      <h1 className="text-center" style={{ color: "#0C234B", fontWeight: "bold" }}>
        The Room Knows
      </h1>
      <p className="text-center text-secondary" style={{ color: "#97999B" }}>
        AI-powered patient exam room
      </p>
      <hr className="mb-4 border-secondary" />

      <div className="d-flex justify-content-end mb-3">
        <button
          onClick={() => router.push("/add")}
          className="btn btn-lg"
          style={{ backgroundColor: "#0C234B", color: "white", borderRadius: "8px" }}
        >
          + Add Patient
        </button>
      </div>

      {loading ? (
        <div className="table-responsive">
          <table className="table table-hover table-bordered">
            <thead style={{ backgroundColor: "#0C234B", color: "white" }}>
              <tr>
                <th>ID</th>
                <th>First Name</th>
                <th>Last Name</th>
                <th>Age</th>
                <th>Last Visit</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 5 }).map((_, index) => (
                <tr key={index} className="placeholder-glow">
                  <td><span className="placeholder col-8"></span></td>
                  <td><span className="placeholder col-6"></span></td>
                  <td><span className="placeholder col-6"></span></td>
                  <td><span className="placeholder col-4"></span></td>
                  <td><span className="placeholder col-6"></span></td>
                  <td>
                    <span className="placeholder col-10"></span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : error ? (
        <p className="text-center mt-5 text-danger">Error fetching data: {error}</p>
      ) : (
        <div className="table-responsive">
          <table className="table table-hover table-bordered">
            <caption className="text-secondary">List of registered patients</caption>
            <thead style={{ backgroundColor: "#0C234B", color: "white" }}>
              <tr>
                <th>ID</th>
                <th>First Name</th>
                <th>Last Name</th>
                <th>Age</th>
                <th>Last Visit</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody style={{ backgroundColor: "#F4F4F4" }}>
              {patientsList.map((patient, index) => (
                <tr key={index}>
                  <td>{patient.patient_id}</td>
                  <td>{patient.patient_first_name}</td>
                  <td>{patient.patient_last_name}</td>
                  <td>{patient.patient_age}</td>
                  <td>{patient.last_visit}</td>
                  <td>
                    <div className="btn-group">
                      <button
                        className="btn btn-sm"
                        style={{ backgroundColor: "#004F9E", color: "white", borderRadius: "5px" }}
                        onClick={() => router.push(`/edit/${patient.patient_id}`)}
                      >
                        Edit
                      </button>
                      <button
                        className="btn btn-sm"
                        style={{ backgroundColor: "#3F72AF", color: "white", borderRadius: "5px" }}
                        onClick={() => router.push(`/newVisit/${patient.patient_id}`)}
                      >
                        New Visit
                      </button>
                      <button
                        className="btn btn-sm"
                        style={{ backgroundColor: "#AB0520", color: "white", borderRadius: "5px" }}
                        onClick={() => router.push(`/delete/${patient.patient_id}`)}
                      >
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default withAuth(Home);
