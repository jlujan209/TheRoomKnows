'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';

const api_key = process.env.NEXT_PUBLIC_API_KEY;


export default function LoginPage() {
  const router = useRouter();
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setCredentials({ ...credentials, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    try {
      const response = await fetch('https://localhost:5000/login', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json', 
            "API-Key": api_key, 
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();

      if (!response.ok) throw new Error(data.error || "Login failed");

      localStorage.setItem('authToken', data.access_token);

      router.push('/home');
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="container mt-4">
      <h1 className="text-center">Login</h1>
      <form onSubmit={handleSubmit} className="mt-3">
        <div className="mb-3">
          <label className="form-label">Username</label>
          <input type="text" className="form-control" name="username" onChange={handleChange} required />
        </div>
        <div className="mb-3">
          <label className="form-label">Password</label>
          <input type="password" className="form-control" name="password" onChange={handleChange} required />
        </div>
        {error && <p className="text-danger">{error}</p>}
        <button type="submit" className="btn btn-primary">Login</button>
      </form>
    </div>
  );
}
