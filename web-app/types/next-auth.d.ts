import NextAuth from "next-auth";

declare module "next-auth" {
  interface Session {
    apiKey?: string;
    user?: {
      id?: string | null;
      name?: string | null;
      email?: string | null;
    };
  }

  interface User {
    apiKey?: string;
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    apiKey?: string;
  }
}
