interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
}

export default function MetricCard({ label, value, unit }: MetricCardProps) {
  return (
    <div
      style={{
        border: "1px solid #ddd",
        borderRadius: 12,
        padding: 16,
        background: "#fff",
      }}
    >
      <div
        style={{
          fontSize: 13,
          color: "#666",
          marginBottom: 8,
        }}
      >
        {label}
      </div>

      <div
        style={{
          fontSize: 24,
          fontWeight: 700,
        }}
      >
        {value}
        {unit && (
          <span
            style={{
              fontSize: 14,
              marginLeft: 4,
              color: "#666",
            }}
          >
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}