'use client';
import { useRouter } from 'next/navigation';
import { patientsList } from "./sampledata/sampledata";
import './global.css';

export default function Home() {
  const router = useRouter();

  return (
    <>
      <h1>The Room Knows</h1>
      <hr></hr>
      <table>
        <caption>Patients</caption>
        <thead>
          <tr>
            <th>ID</th>
            <th>First Name</th>
            <th>Last Name</th>
            <th>Age</th>
            <th>Last Visit</th>
            <th colSpan="2"></th>
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
                <td><button onClick={()=> router.push(`/patient/${patient.patient_id}`)}>Edit</button></td>
                <td><button onClick={()=> router.push('/test')}>New Visit</button></td>
              </tr>)
            })
          }
        </tbody>
      </table>
    </>
  );
}
