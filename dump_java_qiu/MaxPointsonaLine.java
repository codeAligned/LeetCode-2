/**
 * Definition for a point.
 * class Point {
 *     int x;
 *     int y;
 *     Point() { x = 0; y = 0; }
 *     Point(int a, int b) { x = a; y = b; }
 * }
 */

public class Solution {
    public int maxPoints(Point[] points) {
        if (points == null || points.length < 3) {
            return points.length;
        }
        
        HashMap<Double, Integer> map = new HashMap<Double, Integer>();
        double slope = 0.0;
        int max = 1;
        
        for (int i = 0; i < points.length; i++) {
            map.clear();
            int duplicate = 0;
            
            for (int j = i + 1; j < points.length; j++) {

                if (points[i].x == points[j].x && points[i].y == points[j].y) {
                    duplicate++;
                    continue;
                } 
                
                if (points[i].x == points[j].x) {
                    slope = Integer.MAX_VALUE;
                } else {
                    slope = 0.0 + (double)(points[i].y - points[j].y) / (points[i].x - points[j].x);
                }
                
                if (map.containsKey(slope)) {
                    map.put(slope, map.get(slope) + 1);
                } else {
                    map.put(slope, 2);
                }
            }

            int count = 1;
            for (int val : map.values()) {
                count = Math.max(count, val);
            }
            max = Math.max(max, count + duplicate);
        }

        return max;
    }
}
