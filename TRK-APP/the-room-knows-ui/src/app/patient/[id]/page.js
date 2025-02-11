'use client';
import { useParams } from 'next/navigation';  // Import useParams

export default function Test() {
  const { id } = useParams(); // Get the 'id' param directly

  if (!id) {
    return <h1>Loading...</h1>; // Show loading if 'id' is not available
  }

  return (
    <>
      <h1>{id}</h1>
    </>
  );
}
