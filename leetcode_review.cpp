#include <iostream>
#include <vector>
#include <stack>

using namespace std;

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode(int x) : val(x), left(NULL), right(NULL) {
    }
};

struct TreeLinkNode {
    int val;
    TreeLinkNode *left, *right, *next;

    TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {
    }
};


struct ListNode {
    int val;
    ListNode *next;

    ListNode(int x) : val(x), next(NULL) {
    }
};

class Solution {
public:
    int singleNumber(int A[], int n) {
        int res = A[0];
        for (int i = 1; i < n; i++) {
            res = res ^ A[i];
        }
        return res;
    }

    int titleToNumber(string s) {
        int num = 0;
        for (string::iterator it = s.begin(); it != s.end(); ++it) {
            num = num * 26 + (*it - 'A' + 1);
        }
        return num;
    }

    vector<int> preOrderTranverse(TreeNode *root) {
        vector<int> result;

        //same as TreeNode const *p, const TreeNode* p
        const TreeNode *p;

        stack<const TreeNode *> s;

        p = root;
        if (p != nullptr) s.push(p);

        while (!s.empty()) {
            p = s.top();
            s.pop();
            result.push_back(p->val);

            if (p->right != nullptr) s.push(p->right);
            if (p->left != nullptr) s.push(p->right);
        }

        return result;
    }

    vector<int> inorderTraversal(TreeNode *root) {
        vector<int> result;
        const TreeNode *p;
        stack<const TreeNode *> s;

        p = root;

        while (!s.empty() || p != nullptr) {
            if (p != nullptr) {
                s.push(p);
                p = p->left;
            } else {
                p = s.top();
                s.pop();
                result.push_back(p->val);
                p = p->right;
            }
        }
        return result;

    }

    void connect(TreeLinkNode *root) {
        if (root == nullptr) return;

        TreeLinkNode *p, *head;
        p = root;
        head = p;

        while (head->left != nullptr) {
            while (p->next != nullptr) {
                p->left->next = p->right;
                p->right->next = p->next->left;
                p = p->next;
            }
            p->left->next = p->right;
            p = head->left;
            head = head->left;
        }
    }

    bool isSameTree(TreeNode *p, TreeNode *q) {
        if (p != nullptr && q != nullptr) {
            return (p->val == q->val) && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        } else {
            return p == nullptr && q == nullptr;
        }
    }

    vector<vector<int> > levelOrder(TreeNode *root) {
        vector<vector<int>> res;
        levelOrderDFS(0, root, res);
        return res;
    }

    void levelOrderDFS(int level, TreeNode *root, vector<vector<int>> &res) {
        if (!root) return;

        if (level >= res.size()) res.push_back(vector<int>());

        res[level].push_back(root->val);

        levelOrderDFS(level + 1, root->left, res);
        levelOrderDFS(level + 1, root->right, res);
    }

    TreeNode *sortedArrayToBST(vector<int> &num) {

    }

    int searchInsert(int A[], int n, int target) {
        if (n < 1 || target <= A[0]) return 0;

        for (int i = 1; i < n; ++i) {
            if (A[i] >= target) return i;
        }

        if (target > A[n - 1]) return n;
    }

    int evalRPN(vector<string> &tokens) {
        stack<string> s;

        for (auto token : tokens) {
            if (!is_operator(token)) {
                s.push(token);
            }
            else {
                int num2 = stoi(s.top());
                s.pop();
                int num1 = stoi(s.top());
                s.pop();
                int tmp;
                if (token[0] == '+') {
                    tmp = num1 + num2;
                } else if (token[0] == '-') {
                    tmp = num1 - num2;
                } else if (token[0] == '*') {
                    tmp = num1 * num2;
                } else if (token[0] == '/') {
                    tmp = num1 / num2;
                }
                s.push(to_string(tmp));
            }
        }

        return stoi(s.top());
    }

    int maxDepth(TreeNode *root) {
        if (root == nullptr) return 0;
        if (root->left == nullptr && root->right == nullptr) return 1;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }

    int maxProfit(vector<int> &prices) {
        int res = 0;
        for (int i = 1; i < prices.size(); ++i) {
            int diff = prices[i] - prices[i - 1];
            if (diff > 0) res += diff;
        }
        return res;
    }

    bool hasCycle(ListNode *head) {
        ListNode *fast = head, *slow = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
    }

    int maxSubArray(int A[], int n) {
        int result = INT_MIN, f = 0;
        for (int i = 0; i < n; ++i) {
            f = max(f + A[i], A[i]);
            result = max(result, f);
        }
        return result;
    }

    ListNode *deleteDuplicates(ListNode *head) {
        if (!head) return head;
        ListNode dummy(head->val + 1);
        dummy.next = head;

        recur(&dummy, head);
        return dummy.next;
    }

private:
    bool is_operator(const string &op) {
        return op.size() == 1 && string("+-*/").find(op) != string::npos;
    }

    void recur(ListNode *prev, ListNode *cur) {
        if (cur == nullptr) return;

        if (prev->val == cur->val) {
            prev->next = cur->next;
            delete cur;
            recur(prev, prev->next);
        } else {
            recur(prev->next, cur->next);
        }
    }
};


int main() {
    Solution sol = Solution();
    int num = sol.titleToNumber("AA");
    cout << num << endl;
    return 0;
}