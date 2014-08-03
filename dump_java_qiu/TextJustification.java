public class Solution {
    public List<String> fullJustify(String[] words, int L) {
        List<String> result = new ArrayList<String>();
        if (words == null || words.length == 0 || L <= 0) {
            result.add("");
            return result;
        }
        
        int start = 0;
        int end = 0;
        int len = 0;
        boolean isLastLine = true;
        for (end = 0; end < words.length; end++) {
            if (len + words[end].length() + (end - start) > L) {
                String str = composeLine(words, start, end - 1, L, L - len, !isLastLine);
                result.add(str);
                start = end;
                len = 0;
            }
            len += words[end].length();
        }
        
        // Compose the last line
        if (start != end) {
            String str = composeLine(words, start, words.length - 1, L, L - len, isLastLine);
            result.add(str);
        }
        return result;
    }
    
    public String composeLine(String[] words, int start, int end, int L, int spaceLength, boolean lastLine) {
        StringBuffer line = new StringBuffer();
        
        // If current line only contains one word or the last line
        if (start == end || lastLine) {
            for (int i = start; i <= end; i++) {
                line.append(words[i]);
                if (i != end) {
                    line.append(' ');
                }
            }
            appendSpace(line, L - line.length());
            return line.toString();
        }
        
        // Current line contains many words and is not the last line
        int numOfSpace = end - start;
        int averageSpace = spaceLength / numOfSpace;
        int extraSpace = spaceLength % numOfSpace;
        
        for (int i = start; i <= end; i++) {
            line.append(words[i]);
            if (i != end) {
                int curSpace = averageSpace + ((i - start) < extraSpace? 1 : 0);
                appendSpace(line, curSpace);
            }
        }
        return line.toString();
    }
    
    public void appendSpace(StringBuffer buf, int num) {
        for (int i = 0; i < num; i++) {
            buf.append(' ');
        }
    }
}
