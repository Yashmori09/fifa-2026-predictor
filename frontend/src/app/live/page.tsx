import LiveClient from "./LiveClient";
import { getLiveData } from "@/lib/live-data";

// Revalidate the page every 30 minutes (ISR).
// First user after expiry triggers a background refetch; rest get instant cached data.
export const revalidate = 1800;

export default async function LivePage() {
  let data;
  try {
    data = await getLiveData();
  } catch (err) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] px-4 gap-3">
        <p className="text-pink text-sm font-mono">Live data temporarily unavailable</p>
        <p className="text-secondary text-xs text-center max-w-[400px]">
          {err instanceof Error ? err.message : "Unknown error"} · Check back in a few minutes.
        </p>
      </div>
    );
  }
  return <LiveClient data={data} />;
}
