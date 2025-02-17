import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';

export default function withAuth(Component) {
  return function AuthenticatedComponent(props) {
    const router = useRouter();
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
      const token = localStorage.getItem("authToken");

      if (!token) {
        router.push('/login');
      } else {
        setIsAuthenticated(true);
      }
    }, []);

    return isAuthenticated ? <Component {...props} /> : null;
  };
}
