'use client';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import './home.css';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

export default function Home() {
  const router = useRouter();
  const [patientsList, setPatientsList ] = useState([]);
  const [ loading, setLoading ] = useState(true);
  const [ error, setError ] = useState(null);

  useEffect(()=>{
    const fetchPatients = async() => {
      try {
        const response = await fetch('https://localhost:5000/patients/all', {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            "API-Key": api_key
          }
        });
        if (!response.ok) {
          throw new Error(`HTTPS error! Status: ${response.status}`);
        }
        const data = await response.json();
        setPatientsList(data.patients);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    }

    fetchPatients();
  },[]);

  if (loading) return <p>Loading patients...</p>;
  if (error) return <p>Error fetching data: {error}</p>;

  return (
    <>
      <h1>The Room Knows</h1>
      <hr></hr>
      <table>
        <caption>Patients</caption>
        <thead>
          <tr className="headerRow">
            <th>ID</th>
            <th>First Name</th>
            <th>Last Name</th>
            <th>Age</th>
            <th>Last Visit</th>
            <th colSpan="3"><button onClick={()=> router.push('/add')} className="addButton">Add Patient</button></th>
          </tr>
        </thead>
        <tbody>
          {
            patientsList.map((patient, index) => {
              return (
              <tr key={index}>
                <td>{patient.patient_id}</td>
                <td>{patient.patient_first_name}</td>
                <td>{patient.patient_last_name}</td>
                <td>{patient.patient_age}</td>
                <td>{patient.last_visit}</td>
                <td><button onClick={()=> router.push(`/edit/${patient.patient_id}`)}>Edit</button></td>
                <td><button onClick={()=> router.push(`/newVisit/${patient.patient_id}`)}>New Visit</button></td>
                <td><button onClick={()=> router.push(`/delete/${patient.patient_id}`)}>Delete</button></td>
              </tr>)
            })
          }
        </tbody>
      </table>
    </>
  );
}
