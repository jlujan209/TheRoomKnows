'use client';
import { useParams } from 'next/navigation';  
import  ToggleSwitch  from 'the-room-knows-ui/src/app/components/ToggleSwitch';

export default function newVisitPage() {
  const { id } = useParams();

  if (!id) {
    return <h1>Loading...</h1>; 
  }

  return (
    <>
      <h1> New Visit</h1>
      <hr></hr>
      <p>Please select the types of analysis you wish to conduct:</p>
      <form>
        <ToggleSwitch label="Motion Analysis" />
        <ToggleSwitch label="Facial Mapping" />
        <ToggleSwitch label="Emotion Detection" />
        <ToggleSwitch label="Speech Analysis" />
      </form>
    </>
  );
}
