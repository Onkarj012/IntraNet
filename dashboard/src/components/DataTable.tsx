export type Column<T> = {
  key: string;
  header: string;
  align?: "left" | "right";
  render: (row: T) => React.ReactNode;
  className?: string;
};

export default function DataTable<T>({
  columns,
  rows,
  rowKey,
}: {
  columns: Column<T>[];
  rows: T[];
  rowKey?: (row: T) => string;
}) {
  if (!columns?.length) return null;

  const title = columns[0];
  const actions = columns.slice(1).filter((c) => c.header === "");
  const fields = columns.slice(1).filter((c) => c.header !== "");
  const keyOf = (row: T, i: number) => rowKey?.(row) ?? String(i);

  return (
    <>
      {/* Mobile: stacked cards */}
      <div className="flex flex-col gap-3 md:hidden">
        {rows.map((row, i) => (
          <div key={keyOf(row, i)} className="rounded-card border border-hair bg-raised/40 p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="nums text-[15px] font-semibold text-ink">{title.render(row)}</div>
              <div className="flex items-center gap-3">{actions.map((c) => <span key={c.key}>{c.render(row)}</span>)}</div>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-3">
              {fields.map((c) => (
                <div key={c.key} className="flex flex-col gap-1">
                  <span className="text-[10px] font-semibold uppercase tracking-[1.2px] text-muted">{c.header}</span>
                  <span className="nums text-[14px] text-ink">{c.render(row)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Desktop: table */}
      <div className="hidden w-full overflow-x-auto md:block">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  className={`border-b border-line px-3 pb-3 text-[10px] font-semibold uppercase tracking-[1.4px] text-muted ${
                    c.align === "right" ? "text-right" : "text-left"
                  }`}
                >
                  {c.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={keyOf(row, i)} className="group transition-colors hover:bg-raised/50">
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={`nums border-b border-hair px-3 py-3 ${
                      c.align === "right" ? "text-right" : "text-left"
                    } ${c.className ?? ""}`}
                  >
                    {c.render(row)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
