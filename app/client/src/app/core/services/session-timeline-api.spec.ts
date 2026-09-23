import { afterEach, describe, expect, it, vi } from 'vitest';

import { API_BASE_URL } from '../constants';
import { cancelInspectionSessionTimelineJob } from './session-timeline-api';

describe('session timeline API', () => {
  afterEach(() => vi.restoreAllMocks());

  it('requests cancellation through the existing timeline job endpoint', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      JSON.stringify({ job_id: 'job/1', success: true, message: 'Cancellation requested' }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    ));

    const response = await cancelInspectionSessionTimelineJob(42, 'job/1');

    expect(response.success).toBe(true);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
    const [url, init] = fetchSpy.mock.calls[0];
    expect(String(url)).toBe(
      `${API_BASE_URL}/inspection/sessions/42/timeline-jobs/job%2F1`,
    );
    expect(init?.method).toBe('DELETE');
  });
});
