'use client';
import { useParams } from 'next/navigation';  // Import useParams

export default function editPage() {
  const { id } = useParams();

  if (!id) {
    return <h1>Loading...</h1>; 
  }

  return (
    <>
      <h1>Editing: {id}</h1>
    </>
  );
}
