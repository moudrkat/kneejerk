package com.moudrkat.kneejerk;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * A download that continues where it stopped. Bytes go to path + ".part"; every new attempt asks the
 * server for the rest (HTTP Range); the file gets its name only when it is whole. Plain Java, no Android,
 * so it runs in a desktop test as well.
 */
public final class ResumableDownload {
    public interface Progress {
        /** total is -1 when the server did not say; may throw to abort (the partial file stays). */
        void at(long bytes, long total) throws IOException;
    }

    private ResumableDownload() {}

    /** Returns the size of the finished file. Retries dropped connections up to `retries` times in a row. */
    public static long fetch(String url, String path, int retries, Progress progress) throws IOException, InterruptedException {
        final File target = new File(path);
        final File part = new File(path + ".part");
        if (target.exists()) {
            return target.length();
        }
        int failures = 0;
        while (true) {
            long have = part.exists() ? part.length() : 0;
            HttpURLConnection c = null;
            try {
                c = (HttpURLConnection) new URL(url).openConnection();
                c.setConnectTimeout(20000);
                c.setReadTimeout(30000);
                if (have > 0) {
                    c.setRequestProperty("Range", "bytes=" + have + "-");
                }
                final int code = c.getResponseCode();
                if (code == 416) {
                    break; // nothing left to send: the partial file is the whole file
                }
                if (code == 200 && have > 0) {
                    have = 0; // the server ignored the range and sends everything again
                } else if (code != 200 && code != 206) {
                    throw new HttpStatus(code);
                }
                final long rest = c.getContentLengthLong();
                final long total = rest >= 0 ? have + rest : -1;
                try (InputStream in = c.getInputStream();
                     FileOutputStream out = new FileOutputStream(part, have > 0)) {
                    final byte[] buf = new byte[1 << 16];
                    int n;
                    while ((n = in.read(buf)) > 0) {
                        out.write(buf, 0, n);
                        have += n;
                        failures = 0;
                        if (progress != null) {
                            progress.at(have, total);
                        }
                    }
                }
                if (total < 0 || have >= total) {
                    break;
                }
            } catch (Abort | HttpStatus e) {
                throw e;
            } catch (IOException e) {
                if (++failures > retries) {
                    throw e;
                }
                Thread.sleep(Math.min(1000L * failures, 10000L));
            } finally {
                if (c != null) {
                    c.disconnect();
                }
            }
        }
        if (!part.renameTo(target)) {
            throw new IOException("cannot move the finished file into place");
        }
        return target.length();
    }

    /** Thrown by a Progress callback to stop: not a network failure, so it is not retried. */
    public static final class Abort extends IOException {
        public Abort(String message) {
            super(message);
        }
    }

    /** The server answered with something other than the file (404, 403…): retrying will not help. */
    public static final class HttpStatus extends IOException {
        public HttpStatus(int code) {
            super("HTTP " + code);
        }
    }
}
