public class Solution {
    public boolean wordBreak(String s, Set<String> dict) {
        if (s == null || s.length() == 0) {
            return false;
        }
        boolean[] canBreak = new boolean[s.length() + 1];
        canBreak[0] = true;
        
        for (int i = 0; i <= s.length(); i++) {
            if (!canBreak[i]) {
                continue;
            }
            for (String word : dict) {
                int len = word.length();
                int end = i + len;
            
                if (end > s.length()) {
                    continue;
                }
                
                if (canBreak[end]) {
                    continue;
                }
                
                String substr = s.substring(i, end);
                if (canBreak[i] && substr.equals(word)) {
                    canBreak[end] = true;
                }
            }
        }
        return canBreak[s.length()];
    }
}
