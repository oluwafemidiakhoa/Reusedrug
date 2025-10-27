import './globals.css';
import Link from 'next/link';
import { ReactNode } from 'react';
import Providers from './providers';
import AuthStatus from '@/components/AuthStatus';

type Props = {
  children: ReactNode;
};

export default function RootLayout({ children }: Props) {
  return (
    <html lang='en'>
      <body className='min-h-screen bg-slate-50'>
        <Providers>
          <header className='bg-blue-900 py-3 px-4 text-white'>
            <div className='mx-auto flex max-w-5xl flex-col gap-2 md:flex-row md:items-center md:justify-between'>
              <div>
                <h1 className='text-xl font-semibold'>Drug Repurposing Explorer</h1>
                <p className='text-sm text-blue-100'>
                  Accelerate hypotheses with translational science evidence.
                </p>
              </div>
              <div className='flex flex-col items-start gap-2 md:flex-row md:items-center md:gap-4'>
                <nav className='text-sm underline-offset-4'>
                  <Link href='/' className='hover:underline'>
                    Dashboard
                  </Link>
                  <span className='mx-2'>|</span>
                  <Link href='/results' className='hover:underline'>
                    Results
                  </Link>
                  <span className='mx-2'>|</span>
                  <Link href='/workspace' className='hover:underline'>
                    Workspace
                  </Link>
                </nav>
                <AuthStatus />
              </div>
            </div>
          </header>
          <div className='bg-amber-100 px-4 py-2 text-amber-900 shadow'>
            <p className='mx-auto max-w-5xl text-sm'>
              Disclaimer: This MVP provides research insights only and is not medical advice.
              Always consult qualified healthcare professionals before clinical decisions.
            </p>
          </div>
          <main className='mx-auto flex max-w-5xl flex-1 flex-col px-4 py-6'>{children}</main>
        </Providers>
      </body>
    </html>
  );
}

