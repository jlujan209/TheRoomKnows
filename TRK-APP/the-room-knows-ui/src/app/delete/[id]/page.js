'use client';
import { useParams } from 'next/navigation';  // Import useParams

export default function deletePage() {
  const { id } = useParams();

  if (!id) {
    return <h1>Loading...</h1>; 
  }

  return (
    <>
      <h1>Deleting: {id}</h1>
    </>
  );
}
