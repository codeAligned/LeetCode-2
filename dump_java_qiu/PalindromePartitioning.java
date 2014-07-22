public class Solution {
    public List<List<String>> partition(String s) {
        List<List<String>> result = new ArrayList<List<String>>();
        if (s == null || s.length() == 0) {
            return result;
        }
        List<String> answer = new ArrayList<String>();
        helpPartition(s, 0, result, answer);
        return result;
    }
    
    private void helpPartition(String s, int start, List<List<String>> result,
        List<String> answer) {
        if (start == s.length()) {
            result.add(new ArrayList<String>(answer));
            return;
        }
        for (int i = start; i < s.length(); i++) {
            String substr = s.substring(start, i + 1);
            if (isPalindrome(substr)) {
                answer.add(substr);
                helpPartition(s, i + 1, result, answer);
                answer.remove(answer.size() - 1);
            }
        }
    }
    
    private boolean isPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            if (s.charAt(i++) != s.charAt(j--)) {
                return false;
            }
        }
        return true;
    }
}
