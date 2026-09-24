#include <bits/stdc++.h>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

const string INVENTORY_FILE = "inventory.txt";
const string RECORD_FILE = "records.txt";

// ==================== 数据结构 ====================

// 进货/销售记录
struct record{
    int cno;        // 商品编号
    string name;    // 商品名称
    string cat;     // 商品类别（商品删除后仍可据此统计）
    string type;    // 操作类型: buy=进货, sell=销售
    string people;  // 操作人
    string time;    // 操作时间: YYYY-MM-DD_HH:MM:SS
    int quantity;   // 操作数量
};

class commodity{
    public:
        string name;
        string cat;
        double price;
        int no;
        int quantity;
        commodity(string n, string c, double p, int no_, int q)
            : name(n), cat(c), price(p), no(no_), quantity(q){}

    int get_quantity(){
        return quantity;
    }
    double get_price(){
        return price;
    }
    // 进货/销售：库存不足时销售失败，成功返回 true
    bool manage(record &r){
        if(r.type == "buy"){
            quantity += r.quantity;
            cout << "进货成功：" << name << " +" << r.quantity
                 << "，当前库存 " << quantity << endl;
            return true;
        }
        if(r.type == "sell"){
            if(quantity >= r.quantity){
                quantity -= r.quantity;
                cout << "销售成功：" << name << " -" << r.quantity
                     << "，当前库存 " << quantity << endl;
                return true;
            }
            cout << "库存不足：需要 " << r.quantity
                 << "，当前仅有 " << quantity << endl;
        }
        return false;
    }
};

class food : public commodity{
    public:
        food(string n, double p, int no_, int q)
            : commodity(n, "food", p, no_, q){}
};

class electronics : public commodity{
    public:
        electronics(string n, double p, int no_, int q)
            : commodity(n, "electronics", p, no_, q){}
};

// 根据类别创建对应子类对象
commodity* make_commodity(const string &cls, const string &name,
                          double price, int no, int q){
    if(cls == "food") return new food(name, price, no, q);
    if(cls == "electronics") return new electronics(name, price, no, q);
    return new commodity(name, cls, price, no, q);
}

// ==================== 工具函数 ====================

string now_time(){
    time_t t = time(nullptr);
    char buf[32];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H:%M:%S", localtime(&t));
    return buf;
}

// 时间显示：下划线换成空格，便于阅读
string pretty_time(const string &t){
    string s = t;
    replace(s.begin(), s.end(), '_', ' ');
    return s;
}

// 统一时间输入格式：空串=不限；只输日期时补齐时刻，保证字符串比较正确
string norm_time(const string &s, bool is_start){
    if(s.empty()) return s;
    if(s.size() == 10) return s + (is_start ? "_00:00:00" : "_23:59:59");
    return s;
}

string cat_cn(const string &cat){
    if(cat == "food") return "食品";
    if(cat == "electronics") return "电子产品";
    return "其他";
}

// 类别输入既支持英文也支持中文
string cat_to_en(const string &s){
    if(s == "食品" || s == "食物") return "food";
    if(s == "电子产品" || s == "电子") return "electronics";
    return s;
}

string input_line(const string &prompt){
    cout << prompt << flush;
    string s;
    getline(cin, s);
    return s;
}

int input_int(const string &prompt){
    string line = input_line(prompt);
    int x;
    while(!(istringstream(line) >> x))
        line = input_line("输入无效，请输入整数：");
    return x;
}

// ==================== 文件读写 ====================

vector<commodity*> load_inventory(){
    vector<commodity*> v;
    ifstream in(INVENTORY_FILE);
    if(!in){
        cerr << "错误：无法打开 " << INVENTORY_FILE
             << "（实际查找路径：" << fs::absolute(INVENTORY_FILE) << "）" << endl;
        cerr << "请确认该文件存在，且程序是在文件所在目录（当前目录：" << fs::current_path() << "）下运行。" << endl;
        return v;
    }
    string line;
    int lineno = 0;
    while(getline(in, line)){
        lineno++;
        if(line.empty()) continue;      // 跳过空行
        istringstream iss(line);
        string cls, name;
        double price;
        int no, q;
        if(!(iss >> no >> cls >> name >> price >> q)){
            // 首字段不是数字的行（表头/注释）静默跳过
            if(!isdigit((unsigned char)line[0])) continue;
            cerr << "警告：inventory.txt 第 " << lineno
                 << " 行格式不合法，已跳过：" << line << endl;
            continue;
        }
        v.push_back(make_commodity(cls, name, price, no, q));
    }
    return v;
}

void save_inventory(const vector<commodity*> &v){
    ofstream out(INVENTORY_FILE, ios::trunc);
    out << "commodity_no commodity_class commodity_name commodity_price commodity_quantity\n";
    for(auto *c : v)
        out << c->no << " " << c->cat << " " << c->name << " "
            << c->price << " " << c->quantity << "\n";
}

// records.txt 为空/不存在时先写入表头
void ensure_record_header(){
    ifstream in(RECORD_FILE);
    if(in && in.peek() != char_traits<char>::eof())
        return;
    ofstream out(RECORD_FILE, ios::app);
    out << "commodity_no commodity_name commodity_class op_type operator time quantity\n";
}

vector<record> load_records(){
    vector<record> v;
    ifstream in(RECORD_FILE);
    if(!in)
        return v;                       // 尚无记录文件，视为没有记录
    string line;
    int lineno = 0;
    while(getline(in, line)){
        lineno++;
        if(line.empty()) continue;
        istringstream iss(line);
        record r;
        if(!(iss >> r.cno >> r.name >> r.cat >> r.type >> r.people >> r.time >> r.quantity)){
            // 首字段不是数字的行（表头）静默跳过
            if(!isdigit((unsigned char)line[0])) continue;
            cerr << "警告：records.txt 第 " << lineno
                 << " 行格式不合法，已跳过：" << line << endl;
            continue;
        }
        v.push_back(r);
    }
    return v;
}

void append_record(const record &r){
    ensure_record_header();
    ofstream out(RECORD_FILE, ios::app);
    out << r.cno << " " << r.name << " " << r.cat << " " << r.type << " "
        << r.people << " " << r.time << " " << r.quantity << "\n";
}

// ==================== 功能实现 ====================

commodity* find_commodity(const vector<commodity*> &inv, int no){
    for(auto *c : inv)
        if(c->no == no)
            return c;
    return nullptr;
}

// 1. 商品目录查看：按类别组织展示
void show_catalog(const vector<commodity*> &inv){
    if(inv.empty()){
        cout << "（暂无商品）" << endl;
        return;
    }
    map<string, vector<commodity*>> groups;
    for(auto *c : inv)
        groups[c->cat].push_back(c);
    for(auto &[cat, items] : groups){
        cout << "\n【" << cat_cn(cat) << "（" << cat << "）】" << endl;
        for(auto *c : items)
            cout << "  编号 " << c->no << "  名称 " << c->name
                 << "  单价 " << c->price << "  库存 " << c->quantity << endl;
    }
}

// 2/3. 进货与销售
void do_transaction(vector<commodity*> &inv, const string &type){
    string op_cn = (type == "buy" ? "进货" : "销售");
    int no = input_int("商品编号：");
    commodity *c = find_commodity(inv, no);
    if(!c){
        cout << "未找到编号为 " << no << " 的商品" << endl;
        return;
    }
    int qty = input_int(op_cn + "数量：");
    if(qty <= 0){
        cout << "数量必须大于 0" << endl;
        return;
    }
    string people = input_line("操作人：");
    record r{no, c->name, c->cat, type, people, now_time(), qty};
    if(c->manage(r)){                   // 成功才写记录并保存库存
        append_record(r);
        save_inventory(inv);
    }
}

// 4. 删除商品：只从库存删除，进销记录保留
void delete_commodity(vector<commodity*> &inv){
    int no = input_int("要删除的商品编号：");
    for(auto it = inv.begin(); it != inv.end(); ++it){
        if((*it)->no == no){
            string ok = input_line("确认删除 " + (*it)->name + " ？（y/n）");
            if(ok != "y" && ok != "Y"){
                cout << "已取消" << endl;
                return;
            }
            delete *it;
            inv.erase(it);
            save_inventory(inv);
            cout << "已删除，其进货/销售记录仍保留在 " << RECORD_FILE << " 中" << endl;
            return;
        }
    }
    cout << "未找到编号为 " << no << " 的商品" << endl;
}

// 5. 按类别浏览：该类商品按库存量降序展示
void browse_category(const vector<commodity*> &inv){
    string cat = cat_to_en(input_line("类别（food/食品、electronics/电子产品）："));
    vector<commodity*> items;
    for(auto *c : inv)
        if(c->cat == cat)
            items.push_back(c);
    if(items.empty()){
        cout << "该类别下暂无商品" << endl;
        return;
    }
    sort(items.begin(), items.end(), [](const commodity *a, const commodity *b){
        if(a->quantity != b->quantity) return a->quantity > b->quantity;
        return a->no < b->no;
    });
    cout << "\n【" << cat_cn(cat) << "】按库存降序：" << endl;
    for(auto *c : items)
        cout << "  编号 " << c->no << "  名称 " << c->name
             << "  单价 " << c->price << "  库存 " << c->quantity << endl;
}

// 6. 进销记录查询：按编号检索，支持时间范围/操作人筛选
void query_records(){
    vector<record> recs = load_records();
    int no = input_int("商品编号（0=全部商品）：");
    string start = norm_time(input_line("开始时间（YYYY-MM-DD 或 YYYY-MM-DD_HH:MM:SS，回车不限）："), true);
    string end = norm_time(input_line("结束时间（同上，回车不限）："), false);
    string people = input_line("操作人（回车不限）：");

    cout << "\n编号  名称        类型  操作人     时间                数量" << endl;
    cout << string(66, '-') << endl;
    int cnt = 0;
    for(const auto &r : recs){
        if(no != 0 && r.cno != no) continue;
        if(!start.empty() && r.time < start) continue;
        if(!end.empty() && r.time > end) continue;
        if(!people.empty() && r.people != people) continue;
        cout << setw(4) << r.cno << "  " << setw(10) << left << r.name
             << "  " << (r.type == "buy" ? "进货" : "销售")
             << "    " << setw(8) << left << r.people
             << "  " << pretty_time(r.time)
             << "  " << (r.type == "buy" ? "+" : "-") << r.quantity << endl;
        cnt++;
    }
    if(cnt == 0)
        cout << "（没有符合条件的记录）" << endl;
}

// 7. 销量汇总：时间范围内全部/某类商品的总销量
void sales_summary(){
    vector<record> recs = load_records();
    string start = norm_time(input_line("开始时间（YYYY-MM-DD 或 YYYY-MM-DD_HH:MM:SS，回车不限）："), true);
    string end = norm_time(input_line("结束时间（同上，回车不限）："), false);
    string cat = cat_to_en(input_line("类别（回车=全部商品，或 food/食品、electronics/电子产品）："));

    map<int, pair<string, int>> sold;   // 编号 -> {名称, 销量}
    for(const auto &r : recs){
        if(r.type != "sell") continue;
        if(!start.empty() && r.time < start) continue;
        if(!end.empty() && r.time > end) continue;
        if(!cat.empty() && r.cat != cat) continue;
        sold[r.cno].first = r.name;
        sold[r.cno].second += r.quantity;
    }

    cout << "\n销量汇总（" << (cat.empty() ? "全部商品" : cat_cn(cat)) << "）" << endl;
    if(!start.empty() || !end.empty())
        cout << "时间范围：" << (start.empty() ? "不限" : pretty_time(start))
             << " ~ " << (end.empty() ? "不限" : pretty_time(end)) << endl;
    if(sold.empty()){
        cout << "该范围内暂无销售记录" << endl;
        return;
    }
    int total = 0;
    for(auto &[no, p] : sold){
        cout << "  编号 " << no << "  " << p.first << "：销量 " << p.second << endl;
        total += p.second;
    }
    cout << "------------------------------------------" << endl;
    cout << "总销量：" << total << endl;
}

// ==================== 主程序 ====================

int main(){
    cout << "数据文件目录：" << fs::current_path() << endl;
    vector<commodity*> inv = load_inventory();
    ensure_record_header();

    while(true){
        cout << "\n========== 商品管理系统 ==========" << endl;
        cout << "1. 查看商品目录" << endl;
        cout << "2. 进货" << endl;
        cout << "3. 销售" << endl;
        cout << "4. 删除商品" << endl;
        cout << "5. 按类别浏览（按库存排序）" << endl;
        cout << "6. 进销记录查询" << endl;
        cout << "7. 销量汇总" << endl;
        cout << "0. 退出" << endl;
        int op = input_int("请选择操作：");

        if(op == 0) break;
        else if(op == 1) show_catalog(inv);
        else if(op == 2) do_transaction(inv, "buy");
        else if(op == 3) do_transaction(inv, "sell");
        else if(op == 4) delete_commodity(inv);
        else if(op == 5) browse_category(inv);
        else if(op == 6) query_records();
        else if(op == 7) sales_summary();
        else cout << "无效选项，请重新输入" << endl;
    }

    for(auto *c : inv) delete c;
    cout << "已退出，数据已保存。" << endl;
    return 0;
}
