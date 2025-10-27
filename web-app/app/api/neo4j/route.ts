import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8080';

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const endpoint = searchParams.get('endpoint') || 'health';
  const drugId = searchParams.get('drugId');

  try {
    let url = `${BACKEND_URL}/v1/neo4j/${endpoint}`;

    if (drugId && endpoint === 'drug-connections') {
      url = `${BACKEND_URL}/v1/neo4j/drug/${drugId}/connections`;
    }

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Backend responded with ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Neo4j API error:', error);
    return NextResponse.json(
      { error: 'Failed to fetch Neo4j data', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const endpoint = searchParams.get('endpoint') || 'populate';

  try {
    const body = await request.json();

    const response = await fetch(`${BACKEND_URL}/v1/neo4j/${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Backend responded with ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Neo4j API error:', error);
    return NextResponse.json(
      { error: 'Failed to process Neo4j request', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}
