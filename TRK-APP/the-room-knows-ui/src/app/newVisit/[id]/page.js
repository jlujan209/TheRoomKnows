'use client';
import { useParams } from 'next/navigation';  // Import useParams

export default function newVisitPage() {
  const { id } = useParams();

  if (!id) {
    return <h1>Loading...</h1>; 
  }

  return (
    <>
      <h1> New Visit: {id}</h1>
    </>
  );
}
