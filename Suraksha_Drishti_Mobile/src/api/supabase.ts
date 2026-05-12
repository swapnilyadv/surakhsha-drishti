import { createClient } from '@supabase/supabase-js';

// These environment variables should be provided via a .env file or your environment.
const SUPABASE_URL = process.env.SUPABASE_URL || '';
const SUPABASE_PUBLISHABLE_KEY = process.env.SUPABASE_PUBLISHABLE_KEY || '';

if (!SUPABASE_URL || !SUPABASE_PUBLISHABLE_KEY) {
  // In development it's useful to surface a clear error.
  // In production you might want to handle this differently.
  // eslint-disable-next-line no-console
  console.warn('Supabase keys are not set. Authentication will not work until SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY are provided.');
}

export const supabase = createClient(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY);
