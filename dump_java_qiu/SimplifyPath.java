public class Solution {
/*
    '/', skip, find next element
    "//", equals to '/', find next element
    '.' find next element
    '..', pop out the stack, find next element
    else (like ...) push element to the stack, ele -> new top ele, find next element
*/
    public String simplifyPath(String path) {
        if (path == null || path.length() == 0) {
            return "";
        }
        Stack<String> wordStack = new Stack<String>();
        int i = 0;
        while (i < path.length()) {
            while (i < path.length() && path.charAt(i) == '/') {
                i++;
            }
            int start = i;
            while (i < path.length() && path.charAt(i) != '/') {
                i++;
            }
            String substr = path.substring(start, i);
            if (!substr.isEmpty() && !substr.equals(".")) {
                if (substr.equals("..")) {
                    if (!wordStack.isEmpty()) {
                        wordStack.pop();
                    }
                } else {
                    wordStack.push(substr);
                }
            }
        }
        
        String res = new String();
        if (wordStack.isEmpty()) {
            res = "/";
        } else {
            while (!wordStack.isEmpty()) {
                res = "/" + wordStack.pop() + res;
            }
        }
        return res;
    }
}
