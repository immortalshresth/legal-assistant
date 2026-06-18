from retrieval_engine import LegalRetrievalEngine

def inject_real_law():
    db = LegalRetrievalEngine()

    real_ipc_data = [
        {
            "text": "Section 420: Cheating and dishonestly inducing delivery of property. Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine.",
            "metadata": {"section": "Section 420", "act": "Indian Penal Code"}
        },
        {
            "text": "Section 421: Dishonest or fraudulent removal or concealment of property to prevent distribution among creditors. Whoever dishonestly or fraudulently removes, conceals or delivers to any person, or transfers or causes to be transferred to any person, without adequate consideration, any property, intending thereby to prevent the distribution of that property according to law among his creditors, shall be punished with imprisonment of either description for a term which may extend to two years, or with fine, or with both.",
            "metadata": {"section": "Section 421", "act": "Indian Penal Code"}
        }
    ]

    print("Injecting authentic IPC data into persistent storage...")
    db.add_legal_chunks(real_ipc_data)
    print("Database built successfully!")

if __name__ == "__main__":
    inject_real_law()