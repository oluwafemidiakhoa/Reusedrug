'use client';

import { signIn, signOut, useSession } from 'next-auth/react';

export default function AuthStatus() {
  const { data: session, status } = useSession();
  const isAuthenticated = status === 'authenticated';

  return (
    <div className='flex items-center gap-2 text-sm'>
      {isAuthenticated ? (
        <>
          <span className='text-slate-200'>Hi, {session?.user?.name ?? 'workspace user'}</span>
          <button
            onClick={() => signOut({ callbackUrl: '/' })}
            className='rounded border border-slate-200/40 px-2 py-1 text-xs text-white transition hover:bg-white/10'
          >
            Sign out
          </button>
        </>
      ) : (
        <button
          onClick={() => signIn(undefined, { callbackUrl: '/workspace' })}
          className='rounded border border-white/60 px-3 py-1 text-xs font-semibold text-white hover:bg-white/10'
        >
          Sign in
        </button>
      )}
    </div>
  );
}

