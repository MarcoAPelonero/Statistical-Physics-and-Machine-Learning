#pragma once
#include <vector>
#include <random>
#include <deque>
#include <ostream>
#include <fstream>
#include <utility>
#include <cstdlib> // for std::abs
#include <cstdio>  // for std::remove
#include <climits>

#define CRITICAL 4
// Maximum number of animation frames to keep/save for a single avalanche
#define MAX_ANI 10000

class Sandbox {
public:
    struct AvalancheResult; // forward declaration so private members can reference it
private:
private:
    int L;
    std::vector<std::vector<int>> grid;
    mutable std::mt19937 rng;
    mutable std::uniform_int_distribution<int> dist;

    int animationThreshold;             // new: number of topples before we start saving frames
    std::string animationFilePath;      // file path for saved configurations
    bool savingActive = false;          // flag to track if we started saving
    int best_topples = 0;               // number of topples in the best (longest) avalanche seen so far
    int best_step = -1;                 // step index of best avalanche (if known)
    int animationMaxStep = INT_MAX;    // only write animation file for best avalanches with step <= this
    mutable std::vector<std::string> current_frames; // buffer frames for the avalanche currently running
    mutable std::vector<std::string> best_frames;    // stored frames for best avalanche

    inline bool in_bounds(int x, int y) const noexcept {
        return (0 <= x && x < L && 0 <= y && y < L);
    }

    // Save current grid to file (append)
    // Serialize current grid to a string (single frame)
    inline std::string frame_string() const {
        std::string s;
        for (const auto& row : grid) {
            for (int cell : row) { s += std::to_string(cell); s += ' '; }
            s += '\n';
        }
        s += "\n---\n";
        return s;
    }

    // Topple one site once; push any neighbors that become (or remain) critical.
    inline void topple_once(int x, int y, std::deque<std::pair<int,int>>& q, 
                            int& topples, int& affected_accum,
                            int x0, int y0, int& max_dist,
                            int& minx, int& maxx, int& miny, int& maxy) 
    {
        grid[x][y] -= CRITICAL;
        ++topples;

        if (!savingActive && topples >= animationThreshold && animationThreshold > 0) {
            savingActive = true; // start saving frames now
        }

        if (savingActive) {
            // only buffer up to MAX_ANI frames to avoid unbounded memory/disk usage
            if (static_cast<int>(current_frames.size()) < MAX_ANI) {
                current_frames.emplace_back(frame_string());
            }
            // else: silently ignore further frames once the cap is reached
        }

        // update bbox and distance for the site that actually toppled
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
        {
            int d = std::abs(x - x0) + std::abs(y - y0);
            if (d > max_dist) max_dist = d;
        }

        auto try_inc = [&](int nx, int ny) {
            if (in_bounds(nx, ny)) {
                int v = ++grid[nx][ny];
                if (v == CRITICAL) {
                    ++affected_accum;
                    if (nx < minx) minx = nx;
                    if (nx > maxx) maxx = nx;
                    if (ny < miny) miny = ny;
                    if (ny > maxy) maxy = ny;
                    int d = std::abs(nx - x0) + std::abs(ny - y0);
                    if (d > max_dist) max_dist = d;
                }
                if (v >= CRITICAL) q.emplace_back(nx, ny);
            }
        };

        try_inc(x-1, y);
        try_inc(x+1, y);
        try_inc(x, y-1);
        try_inc(x, y+1);
    }

    inline void relax_from(std::deque<std::pair<int,int>>& q, int& topples, int& affected,
                           int x0, int y0, int& duration, int& max_dist,
                           int& minx, int& maxx, int& miny, int& maxy) 
    {
        duration = 0;
        while (!q.empty()) {
            int wave_size = static_cast<int>(q.size());
            ++duration;
            for (int i = 0; i < wave_size; ++i) {
                auto [x, y] = q.front(); 
                q.pop_front();
                if (grid[x][y] >= CRITICAL) {
                    topple_once(x, y, q, topples, affected, x0, y0, max_dist, minx, maxx, miny, maxy);
                } else {
                    if (x < minx) minx = x;
                    if (x > maxx) maxx = x;
                    if (y < miny) miny = y;
                    if (y > maxy) maxy = y;
                    int d = std::abs(x - x0) + std::abs(y - y0);
                    if (d > max_dist) max_dist = d;
                }
            }
        }
    }

public:
    struct AvalancheResult {
        int topples = 0;
        int affected = 0;
        int duration = 0;
        int max_distance = 0;
        int bbox_area = 1;
        int minx = 0, miny = 0, maxx = 0, maxy = 0;
    };

    AvalancheResult best_result;        // metrics for best avalanche

        explicit Sandbox(int L, int animationThreshold = 0, std::string animationFile = "avalanche_frames.txt", int animationMaxStep = INT_MAX)
                : L(L),
                    grid(L, std::vector<int>(L, 0)),
                    rng(std::random_device{}()),
                    dist(0, L - 1),
                    animationThreshold(animationThreshold),
                    animationFilePath(std::move(animationFile)),
                    animationMaxStep(animationMaxStep)
    {
        // ensure stale animation file removed at construction when threshold enabled
        if (this->animationThreshold > 0 && !animationFilePath.empty()) {
            std::remove(animationFilePath.c_str());
        }
    }

    int size() const noexcept { return L; }

    void resetCell(int x, int y) {
        if (in_bounds(x,y)) grid[x][y] = 0;
    }

    bool increaseCell(int x, int y) {
        int v = ++grid[x][y];
        return v >= CRITICAL;
    }

    int getCell(int x, int y) const { return grid[x][y]; }

    AvalancheResult add_and_relax(int x = -1, int y = -1, int stepIndex = -1) {
        if (x < 0 || y < 0) {
            auto rc = selectRandomCell();
            x = rc.first; y = rc.second;
        }
        int topples = 0;
        int affected = 0;
        std::deque<std::pair<int,int>> q;
        int duration = 0;
        int max_dist = 0;
        int minx = x, maxx = x, miny = y, maxy = y;

        // reset animation state
        savingActive = false;
        current_frames.clear();
        if (animationThreshold > 0) {
            // Only clear the on-disk file at construction or when a new best is recorded.
            // We don't touch disk here so multiple avalanches can be compared in-memory.
        }

        if (increaseCell(x, y)) {
            if (grid[x][y] == CRITICAL) ++affected;
            q.emplace_back(x, y);
            relax_from(q, topples, affected, x, y, duration, max_dist, minx, maxx, miny, maxy);
        }

        // If avalanche stops before reaching threshold, delete file
        if (!savingActive && animationThreshold > 0) {
            // nothing buffered and saving never started for this avalanche
        }

        // If this avalanche reached saving threshold, consider it for the best/longest
        if (topples >= animationThreshold && animationThreshold > 0) {
            // If this avalanche is longer than the previously recorded best, replace on-disk file
            if (topples > best_topples) {
                best_topples = topples;
                best_step = stepIndex;
                best_result.topples = topples;
                best_result.affected = affected;
                best_result.duration = duration;
                best_result.max_distance = max_dist;
                best_result.minx = minx; best_result.miny = miny; best_result.maxx = maxx; best_result.maxy = maxy;
                best_result.bbox_area = (affected > 0) ? (maxx - minx + 1) * (maxy - miny + 1) : 0;
                // copy buffered frames, but cap to MAX_ANI
                if (static_cast<int>(current_frames.size()) <= MAX_ANI) {
                    best_frames = current_frames;
                } else {
                    best_frames.assign(current_frames.begin(), current_frames.begin() + MAX_ANI);
                }

                // write best_frames to disk (overwrite) only if we're within allowed animation step
                if (animationFilePath.size() > 0 && (animationMaxStep == INT_MAX || best_step <= animationMaxStep)) {
                    std::ofstream ofs(animationFilePath, std::ios::trunc);
                    int written = 0;
                    for (const auto &f : best_frames) {
                        if (written >= MAX_ANI) break;
                        ofs << f;
                        ++written;
                    }
                    ofs.close();
                }
            }
        }

        AvalancheResult result;
        result.topples = topples;
        result.affected = affected;
        result.duration = duration;
        result.max_distance = max_dist;
        result.minx = minx; result.miny = miny; result.maxx = maxx; result.maxy = maxy;
        result.bbox_area = (affected > 0) ? (maxx - minx + 1) * (maxy - miny + 1) : 0;
        return result;
    }

    AvalancheResult step() { 
        return add_and_relax(-1, -1, -1);
    }

    // step with step index (so sandbox can record which step produced the best avalanche)
    AvalancheResult step(int stepIndex) {
        return add_and_relax(-1, -1, stepIndex);
    }

    std::pair<int, int> selectRandomCell() const {
        return { dist(rng), dist(rng) };
    }

    void saveSandbox(std::ostream &ofs) const {
        for (const auto &row : grid) {
            for (const auto &cell : row) ofs << cell << ' ';
            ofs << '\n';
        }
    }

    // Return best avalanche metrics seen so far
    int getBestTopples() const noexcept { return best_topples; }
    const AvalancheResult& getBestResult() const noexcept { return best_result; }
    int getBestStep() const noexcept { return best_step; }
};
