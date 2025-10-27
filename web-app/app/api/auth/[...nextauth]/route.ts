import NextAuth, { NextAuthOptions } from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';

const AUTH_USERNAME = process.env.AUTH_USERNAME ?? 'admin';
const AUTH_PASSWORD = process.env.AUTH_PASSWORD ?? 'password';
const WORKSPACE_API_KEY = process.env.WORKSPACE_API_KEY ?? '';

export const authOptions: NextAuthOptions = {
  secret: process.env.NEXTAUTH_SECRET,
  session: {
    strategy: 'jwt'
  },
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        username: { label: 'Username', type: 'text' },
        password: { label: 'Password', type: 'password' }
      },
      async authorize(credentials) {
        if (!credentials) {
          return null;
        }
        const { username, password } = credentials as { username: string; password: string };
        if (username === AUTH_USERNAME && password === AUTH_PASSWORD) {
          return {
            id: username,
            name: username,
            apiKey: WORKSPACE_API_KEY
          };
        }
        return null;
      }
    })
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.apiKey = (user as { apiKey?: string }).apiKey ?? '';
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.sub ?? session.user.email ?? 'user';
        (session as unknown as { apiKey?: string }).apiKey = (token as { apiKey?: string }).apiKey ?? '';
      }
      return session;
    }
  }
};

const handler = NextAuth(authOptions);

export { handler as GET, handler as POST };

