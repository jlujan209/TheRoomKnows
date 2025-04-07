"use client";

import { useRouter } from "next/navigation";

const ReportGeneration = ({patient_name}) => {
    const router = useRouter();

    return (
        <>
            <p>Visit Report</p>
            <button onClick={()=> router.push('/home')}>Back to Dashboard</button>
        </>
    );
}

export default ReportGeneration;