'use client';
import { useRouter } from 'next/navigation';
import './global.css';

const api_key = process.env.NEXT_PUBLIC_API_KEY;

export default function Home() {
  let router = useRouter();
 return (
  <>
    <h1>The Room Knows</h1>
    <button type="button" onClick={()=> router.push('/home')}>Get Started</button>
  </>
 );
}
