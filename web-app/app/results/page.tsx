import ResultsWorkspace from "./ResultsWorkspace";

type Props = {
  searchParams: Record<string, string | string[] | undefined>;
};

function parseQuery(value: string | string[] | undefined): string {
  if (Array.isArray(value)) {
    return value[0] ?? "";
  }
  return value ?? "";
}

export default function ResultsPage({ searchParams }: Props) {
  const query = parseQuery(searchParams.q);
  return (
    <div className="flex flex-col gap-6">
      <ResultsWorkspace query={query} />
    </div>
  );
}
