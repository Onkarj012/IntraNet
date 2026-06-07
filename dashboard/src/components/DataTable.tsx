export type Column<T> = {
  key: string;
  header: string;
  align?: "left" | "right";
  render: (row: T) => React.ReactNode;
};

export default function DataTable<T>({
  columns, rows, rowKey,
}: {
  columns: Column<T>[]; rows: T[]; rowKey?: (row: T) => string;
}) {
  if (!columns?.length || !rows?.length) return null;
  const [first, ...rest] = columns;
  const fields = rest.filter((c) => c.header !== "");
  const actions = rest.filter((c) => c.header === "");
  const key = (row: T, i: number) => rowKey?.(row) ?? String(i);

  return (
    <>
      {/* Mobile — stacked cards */}
      <div className="flex flex-col gap-2 sm:hidden">
        {rows.map((row, i) => (
          <div key={key(row, i)} className="card-raised p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-[15px] font-semibold text-ink">{first.render(row)}</span>
              <div className="flex gap-2">{actions.map((c) => <span key={c.key}>{c.render(row)}</span>)}</div>
            </div>
            <div className="grid grid-cols-2 gap-x-5 gap-y-2.5">
              {fields.map((c) => (
                <div key={c.key}>
                  <p className="t-label mb-1">{c.header}</p>
                  <p className="nums text-[13px] text-ink">{c.render(row)}</p>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Desktop — table */}
      <div className="hidden overflow-x-auto sm:block">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c.key}
                  className={`whitespace-nowrap border-b border-hair px-3 pb-3 t-label ${
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
              <tr key={key(row, i)} className="row-hover transition-colors duration-100">
                {columns.map((c) => (
                  <td
                    key={c.key}
                    className={`nums border-b border-hair px-3 py-3 text-ink ${
                      c.align === "right" ? "text-right" : "text-left"
                    }`}
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
